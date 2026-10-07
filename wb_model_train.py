import os, json, math, random, urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_URL = "https://zenodo.org/api/records/14580510/files/Final_data.csv/content"
OUT = "wb_outputs"
os.makedirs(OUT, exist_ok=True)

SEEDS = [42, 52, 62, 72, 82]
LOOKBACK = 4

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def download_data():
    path = os.path.join(OUT, "epiclim_full.csv")
    if not os.path.exists(path):
        urllib.request.urlretrieve(DATA_URL, path)
    return pd.read_csv(path)

def clean_wb_dengue(df):
    df.columns = [c.strip() for c in df.columns]
    df = df[df["state_ut"].astype(str).str.strip().str.lower().eq("west bengal")].copy()
    df["Disease_norm"] = df["Disease"].astype(str).str.strip().str.lower()
    df = df[df["Disease_norm"].eq("dengue")].copy()
    for c in ["Cases","Deaths","preci","LAI","Temp","Latitude","Longitude","day","mon","year"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["mon"], day=df["day"]), errors="coerce")
    df = df.dropna(subset=["date","district","Cases","Latitude","Longitude"]).copy()
    df["district"] = df["district"].astype(str).str.strip()
    return df.sort_values(["date","district"])

def build_panel(df):
    districts = sorted(df["district"].unique())
    # Aggregate duplicate district/week observations.
    g = df.groupby(["date","district"], as_index=False).agg(
        Cases=("Cases","sum"),
        Deaths=("Deaths","sum"),
        Temp=("Temp","mean"),
        preci=("preci","mean"),
        LAI=("LAI","mean"),
        Latitude=("Latitude","mean"),
        Longitude=("Longitude","mean")
    )
    dates = pd.date_range(g.date.min(), g.date.max(), freq="7D")
    # If the source dates are not aligned to a single weekday, use the observed weekly dates.
    observed_dates = pd.DatetimeIndex(sorted(g.date.unique()))
    if len(observed_dates) >= 2:
        diffs = np.diff(observed_dates.values).astype("timedelta64[D]").astype(int)
        if np.nanmedian(diffs) <= 8:
            dates = observed_dates
    idx = pd.MultiIndex.from_product([dates, districts], names=["date","district"])
    p = g.set_index(["date","district"]).reindex(idx).reset_index()
    p["Cases"] = p["Cases"].fillna(0.0)
    p["Deaths"] = p["Deaths"].fillna(0.0)
    for c in ["Temp","preci","LAI","Latitude","Longitude"]:
        p[c] = p.groupby("district")[c].transform(lambda s: s.interpolate().ffill().bfill())
    p["Latitude"] = p["Latitude"].fillna(df["Latitude"].mean())
    p["Longitude"] = p["Longitude"].fillna(df["Longitude"].mean())
    return p, districts

def haversine(lat1, lon1, lat2, lon2):
    r=6371.0
    p1,p2=np.radians(lat1),np.radians(lat2)
    dlat=np.radians(lat2-lat1); dlon=np.radians(lon2-lon1)
    a=np.sin(dlat/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dlon/2)**2
    return 2*r*np.arcsin(np.sqrt(a))

def build_static_graph(panel, districts, k=4):
    coords = panel.groupby("district")[["Latitude","Longitude"]].first().loc[districts]
    n=len(districts)
    D=np.zeros((n,n),dtype=float)
    for i in range(n):
        for j in range(n):
            if i!=j:
                D[i,j]=haversine(coords.iloc[i,0],coords.iloc[i,1],coords.iloc[j,0],coords.iloc[j,1])
    A=np.zeros((n,n),dtype=float)
    sigma=np.median(D[D>0])
    for i in range(n):
        nbr=np.argsort(D[i])[1:k+1]
        for j in nbr:
            A[i,j]=np.exp(-D[i,j]/max(sigma,1e-6))
            A[j,i]=max(A[j,i], A[i,j])
    np.fill_diagonal(A,0)
    return A,D,coords

def dcmg_weights(case_vec, temp_vec, base_A):
    # Direction i -> j. Only t-1 information is used, so there is no same-week leakage.
    x=np.asarray(case_vec,float)
    z=(x-x.mean())/(x.std()+1e-6)
    pressure=1/(1+np.exp(-np.clip(z,-6,6)))
    W=np.zeros_like(base_A)
    for source in range(len(x)):
        for target in range(len(x)):
            if base_A[source,target]>0:
                # Matrix convention: W[target, source] = source -> target.
                W[target,source] = base_A[source,target] * (0.25 + 0.75*pressure[source])
    # Normalize incoming weights at each target.
    rs=W.sum(axis=1,keepdims=True)+1e-8
    W=W/rs
    return W

def build_features(panel,districts):
    dates=sorted(panel.date.unique())
    n=len(districts); t=len(dates)
    arr=np.zeros((t,n,4),float)
    cases=np.zeros((t,n),float)
    lookup={(r.date,r.district):r for r in panel.itertuples(index=False)}
    for ti,d in enumerate(dates):
        for ni,dist in enumerate(districts):
            r=lookup[(d,dist)]
            arr[ti,ni,:]=[r.Cases,r.Temp,r.preci,r.LAI]
            cases[ti,ni]=r.Cases
    return np.array(dates),arr,cases

def make_samples(arr,cases,dates,base_A):
    X=[]; Y=[]; Wseq=[]; target_dates=[]
    for t in range(LOOKBACK,len(dates)):
        X.append(arr[t-LOOKBACK:t])
        Y.append(cases[t])
        ws=[]
        for q in range(t-LOOKBACK,t):
            prev_cases=cases[q-1] if q>0 else cases[q]
            ws.append(dcmg_weights(prev_cases,arr[q,:,1],base_A))
        Wseq.append(np.stack(ws))
        target_dates.append(str(pd.Timestamp(dates[t]).date()))
    return np.asarray(X),np.asarray(Y),np.asarray(Wseq),np.asarray(target_dates)

def norm_split(X,Y,W,dates):
    n=len(X)
    tr=int(n*0.70); va=int(n*0.15)
    # feature normalization from training samples only
    mu=X[:tr].reshape(-1,X.shape[-1]).mean(0)
    sd=X[:tr].reshape(-1,X.shape[-1]).std(0)+1e-6
    Xn=(X-mu)/sd
    # Keep case scale separate for stable target learning.
    ymu=Y[:tr].mean(); ysd=Y[:tr].std()+1e-6
    Yn=(Y-ymu)/ysd
    return Xn,Yn,(mu,sd,ymu,ysd),tr,va

class STGNN(nn.Module):
    def __init__(self, in_dim=4, hidden=32):
        super().__init__()
        self.w1=nn.Linear(in_dim,hidden)
        self.w2=nn.Linear(hidden,hidden)
        self.gru=nn.GRU(hidden,hidden,batch_first=True)
        self.fc=nn.Linear(hidden,1)
    def gcn(self,x,A):
        # x [B,N,F], A [B,N,N]
        I=torch.eye(A.shape[-1],device=A.device).unsqueeze(0)
        S=A+I
        deg=S.sum(-1).clamp_min(1e-6)
        d=deg.rsqrt()
        S=d.unsqueeze(-1)*S*d.unsqueeze(-2)
        h=torch.bmm(S,x)
        h=torch.relu(self.w1(h))
        h=torch.bmm(S,h)
        return torch.relu(self.w2(h))
    def forward(self,x,Aseq):
        hs=[]
        for t in range(x.shape[1]):
            hs.append(self.gcn(x[:,t],Aseq[:,t]))
        h=torch.stack(hs,1) # B,T,N,H
        h=h.permute(0,2,1,3).reshape(x.shape[0]*x.shape[2],x.shape[1],-1)
        h,_=self.gru(h)
        out=self.fc(h[:,-1]).reshape(x.shape[0],x.shape[2])
        return out

def train_model(X,Y,W,tr,va,physics=False,seed=42):
    seed_all(seed)
    device=torch.device("cpu")
    model=STGNN().to(device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    Xtr=torch.tensor(X[:tr],dtype=torch.float32,device=device)
    Ytr=torch.tensor(Y[:tr],dtype=torch.float32,device=device)
    Atr=torch.tensor(W[:tr],dtype=torch.float32,device=device)
    Xva=torch.tensor(X[tr:tr+va],dtype=torch.float32,device=device)
    Yva=torch.tensor(Y[tr:tr+va],dtype=torch.float32,device=device)
    Ava=torch.tensor(W[tr:tr+va],dtype=torch.float32,device=device)
    best=None; bestv=float("inf"); patience=25; wait=0
    for epoch in range(160):
        model.train(); opt.zero_grad()
        pred=model(Xtr,Atr)
        loss=((pred-Ytr)**2).mean()
        if physics:
            # Discrete graph-transmission balance:
            # predicted change should be consistent with incoming lagged source pressure.
            last_cases=Xtr[:,:,-1,0]
            incoming=torch.bmm(Atr[:,-1], torch.relu(last_cases.unsqueeze(-1))).squeeze(-1)
            delta=pred-last_cases
            balance=((delta-(0.15*incoming-0.10*last_cases))**2).mean()
            nonneg=torch.relu(-pred).pow(2).mean()
            loss=loss+0.08*balance+0.02*nonneg
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        model.eval()
        with torch.no_grad():
            pv=model(Xva,Ava); vl=((pv-Yva)**2).mean().item()
        if vl<bestv-1e-5:
            bestv=vl; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; wait=0
        else:
            wait+=1
            if wait>=patience: break
    model.load_state_dict(best)
    return model,epoch+1

def evaluate(model,X,Y,W,yscale):
    with torch.no_grad():
        p=model(torch.tensor(X,dtype=torch.float32),torch.tensor(W,dtype=torch.float32)).numpy()
    pred=p*yscale[1]+yscale[0]; actual=Y*yscale[1]+yscale[0]
    return {
        "MAE":float(mean_absolute_error(actual.ravel(),pred.ravel())),
        "RMSE":float(np.sqrt(mean_squared_error(actual.ravel(),pred.ravel()))),
        "R2":float(r2_score(actual.ravel(),pred.ravel()))
    },pred,actual

def main():
    seed_all(42)
    raw=download_data()
    wb=clean_wb_dengue(raw)
    panel,districts=build_panel(wb)
    dates,arr,cases=build_features(panel,districts)
    A,D,coords=build_static_graph(panel,districts)
    X,Y,W,target_dates=make_samples(arr,cases,dates,A)
    Xn,Yn,norm,tr,va=norm_split(X,Y,W,target_dates)
    mu,sd,ymu,ysd=norm

    panel.to_csv(os.path.join(OUT,"wb_dengue_district_week_panel.csv"),index=False)
    pd.DataFrame({"district":districts,"Latitude":coords.Latitude.values,"Longitude":coords.Longitude.values}).to_csv(os.path.join(OUT,"wb_districts.csv"),index=False)

    edges=[]
    for i,a in enumerate(districts):
        for j,b in enumerate(districts):
            if A[i,j]>0: edges.append({"source":a,"target":b,"base_weight":A[i,j],"distance_km":D[i,j]})
    pd.DataFrame(edges).to_csv(os.path.join(OUT,"wb_static_mobility_graph.csv"),index=False)

    dyn_rows=[]
    for s in range(len(W)):
        for i,a in enumerate(districts):
            for j,b in enumerate(districts):
                if W[s,LOOKBACK-1,i,j]>0:
                    dyn_rows.append({"target_date":target_dates[s],"source":a,"target":b,"weight":float(W[s,LOOKBACK-1,i,j])})
    pd.DataFrame(dyn_rows).to_csv(os.path.join(OUT,"wb_dcmg_laststep_edges.csv"),index=False)

    test_slice=slice(tr+va,None)
    results=[]
    preds_store=[]

    # Persistence baseline
    actual=Y[test_slice]
    persist=X[test_slice,-1,:,0]*ysd+ymu
    act=actual*ysd+ymu
    results.append({"model":"Persistence","MAE":float(mean_absolute_error(act.ravel(),persist.ravel())),
                    "RMSE":float(np.sqrt(mean_squared_error(act.ravel(),persist.ravel()))),
                    "R2":float(r2_score(act.ravel(),persist.ravel()))})

    for physics in [False,True]:
        label="DCMG ST-GNN" if not physics else "Physics-Informed DCMG ST-GNN"
        vals=[]
        for seed in SEEDS:
            model,epochs=train_model(Xn,Yn,W,tr,va,physics,seed)
            met,pred,actual=evaluate(model,Xn[test_slice],Yn[test_slice],W[test_slice],(ymu,ysd))
            met["model"]=label; met["seed"]=seed; met["epochs"]=epochs
            vals.append(met)
            if seed==SEEDS[0]:
                for ti,date in enumerate(target_dates[test_slice]):
                    for ni,d in enumerate(districts):
                        preds_store.append({"date":date,"district":d,"actual_cases":float(actual[ti,ni]),"predicted_cases":float(pred[ti,ni]),"model":label})
        # five-seed aggregate
        results.append({
            "model":label,
            "MAE":float(np.mean([v["MAE"] for v in vals])),
            "RMSE":float(np.mean([v["RMSE"] for v in vals])),
            "R2":float(np.mean([v["R2"] for v in vals])),
            "MAE_std":float(np.std([v["MAE"] for v in vals])),
            "RMSE_std":float(np.std([v["RMSE"] for v in vals])),
            "R2_std":float(np.std([v["R2"] for v in vals]))
        })

    pd.DataFrame(preds_store).to_csv(os.path.join(OUT,"wb_test_predictions_seed42.csv"),index=False)
    pd.DataFrame(results).to_csv(os.path.join(OUT,"wb_model_results.csv"),index=False)

    summary={
        "target":"Dengue",
        "state":"West Bengal",
        "source":"EpiClim Zenodo 10.5281/zenodo.14580510",
        "raw_west_bengal_dengue_rows":int(len(wb)),
        "districts":int(len(districts)),
        "district_names":districts,
        "weeks":int(len(dates)),
        "supervised_samples":int(len(X)),
        "lookback_weeks":LOOKBACK,
        "train_samples":int(tr),
        "validation_samples":int(va),
        "test_samples":int(len(X)-tr-va),
        "static_graph_directed_edges":int((A>0).sum()),
        "dcmg":"Dynamic lagged transmission-pressure graph on a distance-decay mobility proxy; weights use source incidence at t-1 only.",
        "physics":"Discrete graph-transmission balance penalty plus non-negativity penalty; this is not a full SIR parameter-identification model.",
        "results":results
    }
    with open(os.path.join(OUT,"wb_model_summary.json"),"w") as f: json.dump(summary,f,indent=2)

    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
