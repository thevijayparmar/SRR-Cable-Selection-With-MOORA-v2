#!/usr/bin/env python3
# ================================================================
#  Stress-Ribbon Bridge Cable Selector  – Streamlit edition
#  • Dual input (Manual / CSV)
#  • CSV-driven MOORA settings
# ----------------------------------------------------------------
#  Authors : Vijaykumar Parmar & Dr. K. B. Parikh   (© 2025)
# ================================================================

import math, os
from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
import plotly.express as px
import plotly.graph_objects as go # Added for more granular plot control
import streamlit as st

# ---------------------------------------------------------------
CREDIT = "Authors : Vijaykumar Parmar & Dr. K. B. Parikh"

# ---------------- MOORA-criterion dataclass --------------------
@dataclass
class CriterionConfig:
    enabled: bool
    is_cost: bool          # True = cost, False = benefit
    shape: str             # "linear" | "exponential"
    trigger: str           # "above" | "below"
    threshold: float
    slope: float
    exponent: float

DEFAULT_CRIT: Dict[str, CriterionConfig] = {
    "Utilisation"   : CriterionConfig(True,  True , "exponential", "below", 0.8  , 1.0, 6.0),
    "Slope_pct"     : CriterionConfig(True,  False, "linear"     , "below", 2.5  , 1.0, 1.0),
    "Cable_Dia_mm"  : CriterionConfig(True,  True , "linear"     , "above", 150  , 0.5, 1.0),
    "N_Cables"      : CriterionConfig(True,  True , "exponential", "above", 5    , 1.0, 1.2),
    "NatFreq_Hz"    : CriterionConfig(True,  False, "linear"     , "above", 2.0  , 1.0, 1.0),
    "Tension_kN"    : CriterionConfig(True,  True , "linear"     , "above", 0.0  , 1.0, 1.0),
    "Sag_m"         : CriterionConfig(True,  False, "exponential", "below", 0.003, 1.0, 3.0),
}

# --------------- Numeric-range checks for CSV core fields -------
CSV_RANGES = {
    "Span L (m)"       : (10, 500),
    "UDL w (kN/m)"     : (10, 1000),
    "Bridge width (m)" : (1, 10),
    "Base Cable Diam"  : (5, 300),
    "Base Cables"      : (2, 20),
    "Strength (MPa)"   : (200, 3000),
    "Density (kN/m3)"  : (50, 90),
    "Δ cables"         : (0, 5),
}

# ------------------------ CSV template --------------------------
def _template_rows() -> List[Dict[str, Any]]:
    row = {
        "Bridge Name"       : "Demo-Bridge",
        "Span L (m)"        : 80,
        "UDL w (kN/m)"      : 15,
        "Bridge width (m)"  : 3,
        "Base Cable Diam"   : 30,
        "Base Cables"       : 4,
        "Strength (MPa)"    : 1860,
        "Density (kN/m3)"   : 77,
        "Δ cables"          : 1,
    }
    for crit, cfg in DEFAULT_CRIT.items():
        row[f"{crit} Enabled"]   = int(cfg.enabled)
        row[f"{crit} Type"]      = "Cost" if cfg.is_cost else "Benefit"
        row[f"{crit} Shape"]     = cfg.shape
        row[f"{crit} Trigger"]   = cfg.trigger
        row[f"{crit} Threshold"] = cfg.threshold
        row[f"{crit} Slope"]     = cfg.slope
        row[f"{crit} Exponent"]  = cfg.exponent
    return [row]

if not os.path.exists("template_input.csv"):
    pd.DataFrame(_template_rows()).to_csv("template_input.csv", index=False)

# ===============================================================
# ----------  Engineering, design-space & MOORA logic  ----------
# ===============================================================
def _area_mm2(d_mm: float) -> float:
    return math.pi * (d_mm / 2) ** 2

def cable_metrics(span_m, udl_kNpm, n_cables, dia_mm,
                  strength_MPa, utilisation, density_kNpm3):
    A_mm2   = _area_mm2(dia_mm)
    H_kN    = n_cables * A_mm2 * utilisation * strength_MPa / 1_000
    sag_m   = udl_kNpm * span_m**2 / (8 * H_kN) if H_kN else 0
    V_kN    = udl_kNpm * span_m / 2
    T_kN    = math.hypot(H_kN, V_kN)
    A_m2    = A_mm2 * 1e-6
    rho     = density_kNpm3 * 1_000 / 9.81
    mu_kgpm = rho * A_m2
    omega2  = (H_kN * 1_000) / (mu_kgpm * n_cables) if mu_kgpm and n_cables else 0
    f_nat   = (1/(2*span_m))*math.sqrt(omega2) if omega2 else 0
    mass_kg = mu_kgpm * span_m * n_cables
    return {
        "Cable_Dia_mm": dia_mm,
        "Utilisation" : utilisation,
        "N_Cables"    : n_cables,
        "Slope_pct"   : sag_m/span_m*100,
        "Tension_kN"  : T_kN,
        "Sag_m"       : sag_m,
        "NatFreq_Hz"  : f_nat,
        "CableMass_kg": mass_kg,
    }

def _pb_value(x: float, cfg: CriterionConfig) -> float:
    if not cfg.enabled: return 0.0
    diff = (x - cfg.threshold) if cfg.trigger=="above" else (cfg.threshold - x)
    if diff <= 0: return 0.0
    return cfg.slope*diff if cfg.shape=="linear" else math.exp(cfg.exponent*diff)-1

def generate_alternatives(span, udl, base_n, base_d, strength, density,
                          width, util_grid, dia_factors, n_delta):
    recs=[]
    n_opts = [max(2, base_n+i) for i in range(-n_delta, n_delta+1)]
    for fac in dia_factors:
        dia = round(base_d*(1+fac),3)
        if dia<5: continue
        for util in util_grid:
            for n in n_opts:
                r = cable_metrics(span, udl, n, dia, strength, util, density)
                r["Cable_Spacing_m"]   = width/n
                r["UDL_perCable_kNpm"] = udl/n
                recs.append(r)
    return pd.DataFrame(recs).round(6)

def moora_rank(df: pd.DataFrame, cfg_map: Dict[str, CriterionConfig]) -> pd.DataFrame:
    benefit,cost=[],[]
    for crit,cfg in cfg_map.items():
        if not cfg.enabled: continue
        col=f"PB_{crit}"
        df[col]=df[crit].apply(lambda v:_pb_value(v,cfg))
        (cost if cfg.is_cost else benefit).append(col)
    for col in benefit+cost:
        norm=np.sqrt((df[col]**2).sum())
        df[f"N_{col}"]=df[col]/norm if norm else 0
    df["MOORA_Score"]=df[[f"N_{c}" for c in benefit]].sum(axis=1)\
                     -df[[f"N_{c}" for c in cost]].sum(axis=1)
    ranked=df.sort_values("MOORA_Score",ascending=False).reset_index(drop=True)
    ranked.index+=1
    ranked["Rank"]=ranked.index
    return ranked

# ===============================================================
# ----------------------  Plot helpers  -------------------------
# ===============================================================
def cable_profile_fig(span: float, sag: float, equal_scale: bool = False) -> Figure:
    """
    Generates the cable profile plot.
    A cable under UDL hangs in a perfect parabola. It's the platonic ideal
    of structural forms - pure, simple, ruthlessly efficient.
    """
    x = np.linspace(0, span, 200)
    y = -4 * sag * (x / span) * (1 - x / span)
    fig = Figure(figsize=(6, 3))
    ax = fig.add_subplot(111)

    # MODIFICATION: Remove whitespace padding for a tighter plot.
    ax.margins(0)
    
    ax.plot(x, y, color="tab:blue")
    ax.set_xlabel("Span position (m)")
    ax.set_ylabel("Elevation (m, downward)")
    ax.set_title("Cable elevation profile")
    ax.grid(alpha=.3, ls="--")

    # MODIFICATION: Optional equal scaling for X and Y axes.
    # Sometimes, a true-to-scale view is enlightening. Other times, it's just flat.
    if equal_scale:
        ax.set_aspect('equal', adjustable='box')

    fig.text(.5, -.12, CREDIT, ha="center", fontsize=8)
    return fig

def contour_fig(df: pd.DataFrame, xvar:str, yvar:str) -> Figure:
    xi=np.linspace(df[xvar].min(),df[xvar].max(),120)
    yi=np.linspace(df[yvar].min(),df[yvar].max(),120)\
        if yvar!="N_Cables" else np.array(sorted(df[yvar].unique()))
    Xi,Yi=np.meshgrid(xi,yi)
    Zi=griddata((df[xvar],df[yvar]),df["MOORA_Score"],(Xi,Yi),method="cubic")
    cmap=LinearSegmentedColormap.from_list(
        "custom",["black","#8B0000","purple","blue","skyblue","lightgreen","yellow"],256)
    fig=Figure(figsize=(6,4))
    ax=fig.add_subplot(111)
    cs=ax.contourf(Xi,Yi,Zi,levels=15,cmap=cmap)
    ax.set_xlabel(xvar); ax.set_ylabel(yvar)
    ax.set_title("MOORA score contour")
    fig.colorbar(cs,ax=ax,label="MOORA Score")
    fig.text(.5,-.08,CREDIT,ha="center",fontsize=8)
    if yvar=="N_Cables": ax.set_yticks(yi)
    return fig

def parallel_fig(df: pd.DataFrame):
    """
    Generates a parallel coordinates plot for the top alternatives.
    This plot is our window into the high-dimensional chaos of the design space.
    We're trying to tame it by highlighting the champions (top 3)
    while the others fade into a ghostly Greek chorus.
    """
    # MODIFICATION: Filter to only the top 10 alternatives for clarity.
    top10 = df.head(10).copy()

    dimensions = ["Cable_Dia_mm", "Utilisation", "N_Cables", "NatFreq_Hz",
                  "Sag_m", "Tension_kN", "CableMass_kg", "MOORA_Score"]
    
    # MODIFICATION: Rebuilt from the ground up using go.Figure for layering control.
    # TODO: Maybe add a toggle for which dimensions to show? Could get busy.
    fig = go.Figure()

    # Plot ranks 4-10 first as a muted background.
    df_others = top10[top10['Rank'] > 3]
    if not df_others.empty:
        fig.add_trace(go.Parcoords(
            line=dict(color='#D3D3D3', width=1), # Thin grey lines for the runners-up.
            dimensions=[dict(range=[top10[col].min(), top10[col].max()],
                             label=col, values=df_others[col]) for col in dimensions]
        ))

    # Layer the top 3 ranks on top with distinct, bold colors. A visual hierarchy to celebrate the victors.
    colors = {1: 'yellow', 2: 'green', 3: 'blue'}
    for rank in sorted(colors.keys(), reverse=True): # Plot 3, then 2, then 1, so rank 1 is on top
        df_rank = top10[top10['Rank'] == rank]
        if not df_rank.empty:
            fig.add_trace(go.Parcoords(
                line=dict(color=colors[rank], width=4), # Bold lines for the winners.
                dimensions=[dict(range=[top10[col].min(), top10[col].max()],
                                 label=col, values=df_rank[col]) for col in dimensions]
            ))

    # MODIFICATION: Update layout with new title and font size.
    fig.update_layout(
        title="Parallel coordinates – top 10 alternatives",
        font=dict(size=12)
    )
    
    fig.add_annotation(text=CREDIT, x=.5, y=-.12, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=10))
    return fig

# ===============================================================
# ------------------------- STREAMLIT ---------------------------
# ===============================================================
st.set_page_config("SRB Cable Selector – MOORA", layout="wide")
st.title("Stress-Ribbon Bridge Cable Selector (MOORA)")

# Session placeholders
for k in ["span","udl","width","base_n","base_d",
          "strength","density","n_delta","bridge_id","cfg_map"]:
    st.session_state.setdefault(k,None)

# ---------------------   SIDEBAR  ------------------------------
with st.sidebar:
    st.header("Input mode")
    mode=st.radio("Choose input method",["Manual Input","CSV Input"])

    # ----------- MANUAL -----------------
    if mode=="Manual Input":
        st.subheader("Bridge parameters – manual")
        st.session_state.span   = st.number_input("Span L (m)",       10.,500.,50.)
        st.session_state.udl    = st.number_input("UDL w (kN/m)",     10.,1000.,100.)
        st.session_state.width  = st.number_input("Bridge width (m)", 1.,10.,3.)
        st.session_state.base_n = st.number_input("Base number of cables",2,20,2)
        st.session_state.base_d = st.number_input("Base cable diameter (mm)",5.,300.,20.)
        st.session_state.strength=st.number_input("Cable strength σ (MPa)",200.,3000.,1600.)
        st.session_state.density =st.number_input("Density γ (kN/m³)",50.,90.,77.)
        st.session_state.n_delta =st.slider("± range around base #Cables",0,5,1)
        st.session_state.bridge_id=None

        st.markdown("---")
        st.subheader("MOORA criterion settings")
        cfg={}
        for name,def_cfg in DEFAULT_CRIT.items():
            with st.expander(name,expanded=False):
                en = st.checkbox("Enabled",def_cfg.enabled,key=name+"en")
                ct = st.radio("Type",["Cost","Benefit"],0 if def_cfg.is_cost else 1,key=name+"ct")
                sh = st.selectbox("Shape",["linear","exponential"],
                                  0 if def_cfg.shape=="linear" else 1,key=name+"sh")
                tr = st.selectbox("Trigger",["above","below"],
                                  0 if def_cfg.trigger=="above" else 1,key=name+"tr")
                th = st.number_input("Threshold",value=def_cfg.threshold,key=name+"th")
                sl = st.number_input("Slope (linear)",value=def_cfg.slope,key=name+"sl")
                ex = st.number_input("Exponent (exp)",value=def_cfg.exponent,key=name+"ex")
            cfg[name]=CriterionConfig(en,ct=="Cost",sh,tr,th,sl,ex)
        st.session_state.cfg_map=cfg

    # ----------- CSV --------------------
    else:
        st.subheader("Bridge parameters – CSV")
        with open("template_input.csv","rb") as tf:
            st.download_button("📥 Download sample template.csv",tf,
                               file_name="template_input.csv",mime="text/csv")
        up=st.file_uploader("Upload single-row CSV",type="csv")
        if up:
            df=pd.read_csv(up)
            def _style(row):
                return["background-color:red" if col in CSV_RANGES and not(
                       CSV_RANGES[col][0]<=val<=CSV_RANGES[col][1]) else""
                       for col,val in row.items()]
            st.dataframe(df.style.apply(_style,axis=1),height=260)
            idx=st.number_input("Row index to load (starting at 2)",2,len(df)+1,2,step=1)
            if st.button("Load input"):
                try: row=df.iloc[idx-2]
                except IndexError: st.error("Row out of range.")
                else:
                    st.session_state.span      = float(row["Span L (m)"])
                    st.session_state.udl       = float(row["UDL w (kN/m)"])
                    st.session_state.width     = float(row["Bridge width (m)"])
                    st.session_state.base_d    = float(row["Base Cable Diam"])
                    st.session_state.base_n    = int(row["Base Cables"])
                    st.session_state.strength  = float(row["Strength (MPa)"])
                    st.session_state.density   = float(row["Density (kN/m3)"])
                    st.session_state.n_delta   = int(row["Δ cables"])
                    st.session_state.bridge_id = str(row["Bridge Name"])
                    cfg={}
                    for crit,def_cfg in DEFAULT_CRIT.items():
                        cfg[crit]=CriterionConfig(
                            bool(row.get(f"{crit} Enabled",def_cfg.enabled)),
                            str(row.get(f"{crit} Type",
                                "Cost" if def_cfg.is_cost else "Benefit")).lower()=="cost",
                            str(row.get(f"{crit} Shape",def_cfg.shape)).lower(),
                            str(row.get(f"{crit} Trigger",def_cfg.trigger)).lower(),
                            float(row.get(f"{crit} Threshold",def_cfg.threshold)),
                            float(row.get(f"{crit} Slope",def_cfg.slope)),
                            float(row.get(f"{crit} Exponent",def_cfg.exponent))
                        )
                    st.session_state.cfg_map=cfg
                    st.success(f"Loaded row {idx} for **{st.session_state.bridge_id}**")

    run=st.button("Run analysis",type="primary")

# -------------------- ANALYSIS -------------------------
if run:
    req=["span","udl","width","base_n","base_d","strength",
         "density","n_delta","cfg_map"]
    if any(st.session_state[k] is None for k in req):
        st.error("❗ Complete inputs first.")
    else:
        util=[0.6,0.7,0.8,0.9,0.95,0.99,1.0]
        dia_fac=np.linspace(-.5,.5,11)
        alts=generate_alternatives(
            st.session_state.span,st.session_state.udl,
            st.session_state.base_n,st.session_state.base_d,
            st.session_state.strength,st.session_state.density,
            st.session_state.width,util,dia_fac,st.session_state.n_delta
        )
        ranked=moora_rank(alts.copy(),st.session_state.cfg_map)
        st.session_state.ranked_df=ranked
        st.session_state.results_ready=True

# -------------------- RESULTS --------------------------
if st.session_state.get("results_ready"):
    ranked=st.session_state.ranked_df
    best=ranked.iloc[0]
    title="### Preferred alternative"
    if st.session_state.bridge_id:
        title+=f" for **{st.session_state.bridge_id}**"
    st.markdown(
        f"{title}  \n"
        f"* Diameter : **{best.Cable_Dia_mm:.1f} mm** \n"
        f"* Cables  : **{int(best.N_Cables)}** \n"
        f"* Utilisation : **{best.Utilisation:.2f}** \n"
        f"* MOORA score : **{best.MOORA_Score:.3f}** \n\n"
        f"**{CREDIT}**"
    )
    st.table(pd.DataFrame({
        "Parameter":["Span","UDL","Bridge width","Base cables",
                     "Base diameter","Strength","Density"],
        "Value":[st.session_state.span,st.session_state.udl,st.session_state.width,
                 st.session_state.base_n,st.session_state.base_d,
                 st.session_state.strength,st.session_state.density],
        "Unit":["m","kN/m","m","","mm","MPa","kN/m³"],
    }))
    tab1,tab2,tab3=st.tabs(["Cable profile & contour","Parallel plot","Full table"])

    with tab1:
        # MODIFICATION: Updated function call to include the new 'equal_scale' parameter.
        st.pyplot(cable_profile_fig(st.session_state.span, best.Sag_m, equal_scale=False))
        vars=["Utilisation","Cable_Dia_mm","N_Cables","NatFreq_Hz",
              "Sag_m","Tension_kN","CableMass_kg"]
        c1,c2,_=st.columns([3,3,1])
        xsel=c1.selectbox("X variable",vars,key="xsel")
        ysel=c2.selectbox("Y variable",vars,index=1,key="ysel")
        colA,colB=st.columns([1,2])
        if colA.button("Generate"):
            if xsel==ysel: st.warning("Select two different variables."); st.stop()
            st.pyplot(contour_fig(ranked,xsel,ysel))
        if colB.button("Generate All Charts"):
            for i,x in enumerate(vars):
                for j,y in enumerate(vars):
                    if i>=j: continue
                    st.subheader(f"Contour: {x} vs {y}")
                    st.pyplot(contour_fig(ranked,x,y))

    with tab2:
        st.plotly_chart(parallel_fig(ranked),use_container_width=True)

    with tab3:
        st.dataframe(ranked)
        st.download_button("Download CSV",ranked.to_csv(index=False).encode(),
                           file_name="srb_results.csv",mime="text/csv")
else:
    st.info("Set parameters (manual or CSV) and click **Run analysis**.")
