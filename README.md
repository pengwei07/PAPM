# PAPM: A Physics-aware Proxy Model for Process Systems

Process systems, which play a fundamental role in various scientific and engineering fields, often rely on computational models to capture their complex temporal-spatial dynamics.
However, these models struggle to fully encompass underlying intricate physical principles, leading to computational inefficiency and potential inaccuracies.
To address these challenges, we propose a **p**hysics-**a**ware **p**roxy **m**odel (**PAPM**) to explicitly incorporate partial prior mechanistic knowledge, including conservation and constitutive relations.
Additionally, to enhance the inductive biases about strict physical laws and broaden the applicability scope, we introduce a holistic temporal and spatial stepping method (TSSM) aligned with the distinct equation characteristics of different process systems, resulting in better out-of-sample generalization.
We systematically compared state-of-the-art pure data-driven models and physics-aware models, spanning five two-dimensional non-trivial benchmarks in seven generalization tasks. 
Through such analysis, the structural design and specialized spatio-temporal modeling schemes (i.e., TSSM) of PAPM demonstrated not only notable training/inference efficiency and boosted accuracies, but also an impressive out-of-sample generalization.
Notably, PAPM achieved an average absolute performance boost of 6.4\%, requiring fewer FLOPs, and utilizing only 1\%-10\% of the parameters compared to alternative methods.

**The core contributions of this work are:**
- The proposal of PAPM, a novel physics-aware architecture design that explicitly incorporates partial prior mechanistic knowledge such as BCs, conservation, and constitutive relations. This design proves to be superior in terms of both training efficiency and out-of-sample generalizability.
- The introduction of TSSM, a holistic spatio-temporal stepping modeling method. It aligns with the distinct equation characteristics of different process systems by employing stepping schemes via temporal and spatial operations, whether in physical or spectral space.
- A systematic evaluation of state-of-the-art pure data-driven models alongside physics-aware models, spanning five two-dimensional non-trivial benchmarks in nine generalization tasks, as depicted in Fig.~\ref{first}(c). Notably, PAPM achieved an average absolute performance boost of 6.4\%, requiring fewer FLOPs, and utilizing only 1\%-10\% of the parameters compared to alternative methods.
  
## Data

Dataset Link:
- 1. Burger2d (from PDENet++'s experiments): [Burger2d]()
- 2. RD2d (from PDEBench's experiments): [RD2d](https://darus.uni-stuttgart.de/file.xhtml?fileId=133017&version=5.0)
- 3. NS2d (from FNO's experiments):  [NS2d](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-)

FNO

https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-
