# CircuitMind Eval v0.2.0 — Solution Notes

## CM001 — Series (easy)
Req=4+8=12Ω; I=12/12=1A; drops: 4V and 8V.

## CM002 — Parallel (easy)
Req=6||3=2Ω; Itotal=6A; branch currents: 2A and 4A; each branch has 12V.

## CM003 — User-provided multi-source circuit (medium)
Bottom rail=0V. V1 makes its top node −80V; V3 makes the node below R2 +40V; V2 imposes Va−Vb=100V. Supernode KCL: (Va+80)/2+(Va−40)/12+Vb/4=0. Thus Va=−14V, Vb=−114V. Downward R3 current is −28.5A (actual 28.5A upward). R3 absorbs P=114²/4=3249W.

## CM004 — Nodal analysis (medium)
3kΩ||6kΩ=2kΩ; total 4kΩ; source current 2.5mA; Va=5V; branch currents 1.6667mA and 0.8333mA.

## CM005 — Thevenin (medium)
Vth=18V; deactivate source by shorting it, giving Rth=3Ω. With RL=6Ω, IL=2A and VL=12V.

## CM006 — Superposition (hard)
With V1 alone: (Va−12)/2+Va/3+Va/6=0, so Va1=6V. With V2 alone: Va/2+(Va−6)/3+Va/6=0, so Va2=2V. Total Va=8V; IL=8/6=1.3333A.

## CM007 — Norton (medium)
IN=24/6=4A; RN=6Ω; with RL=3Ω, IL=4×6/(6+3)=8/3A and VL=8V.

## CM008 — Mesh (hard)
Equivalent node equation: (Va−12)/2+(Va−6)/2+Va/4=0, so Va=7.2V. Left current 2.4A toward Va; right current 0.6A from Va toward +6V; shared current 1.8A downward.

## CM009 — Maximum power transfer (medium)
Rth=6Ω, so RL=6Ω. Current=24/12=2A; VL=12V; Pmax=24W.

## CM010 — GATE-level bridge (hard)
KCL: (VL−20)/2+VL/4+(VL−VR)/3=0; (VR−20)/4+VR/2+(VR−VL)/3=0. VL=200/17=11.7647V; VR=140/17=8.2353V. Bridge current left-to-right=20/17=1.1765A. Source current=120/17=7.0588A.
