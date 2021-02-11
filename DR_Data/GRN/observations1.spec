// cell SHSY5Y; 120 h; dose mi666; perturbagen CACNA1C (trt_sh)
#Experiment1[0] |= $KnockDown1;
#Experiment1[0] |= $Initial1;
#Experiment1[18] |= $Final1;
#Experiment1[19] |= $Final1;
fixpoint(#Experiment1[19]);

// cell SHSY5Y; 120 h; dose mi666; perturbagen CDC42 (trt_sh)
#Experiment2[0] |= $KnockDown2;
#Experiment2[0] |= $Initial2;
#Experiment2[18] |= $Final2;
#Experiment2[19] |= $Final2;
fixpoint(#Experiment2[19]);

// cell SHSY5Y; 120 h; dose mi666; perturbagen KCNA2 (trt_sh)
#Experiment3[0] |= $KnockDown3;
#Experiment3[0] |= $Initial3;
#Experiment3[18] |= $Final3;
#Experiment3[19] |= $Final3;
fixpoint(#Experiment3[19]);

$KnockDown1 :=
{ KO(SOD1) = 0 and KO(KCNA2) = 0 and KO(CACNA1C) = 1 and KO(SYT1) = 0 and KO(CDC42) = 0
};

$Initial1 :=
{ ADAM23 = 1 and AMPH = 1 and AP3B2 = 0 and ARHGAP44 = 0 and ARL3 = 0 and ATP1A3 = 1 and ATP6V1A = 1 and CACNA1C = 0 and CACNB2 = 0 and CACNB4 = 1 and CACNG2 = 1 and CADPS = 0 and CAMK2B = 1 and CCNB1 = 0 and CDC42 = 0 and CDK14 = 1 and CHML = 1 and CIT = 0 and CKS2 = 1 and CNTNAP1 = 0 and CYP2C8 = 1 and DNM1 = 1 and EEF1A2 = 0 and ENO2 = 1 and GAP43 = 0 and GFPT1 = 1 and GGH = 1 and GPI = 0 and GRIN1 = 1 and INA = 1 and KCNA1 = 0 and KCNA2 = 1 and KCNAB2 = 1 and MAP3K9 = 1 and MAPK8 = 1 and MCF2 = 0 and MYO1B = 1 and MYO5A = 0 and NEFH = 0 and NEFM = 0 and NELL2 = 1 and PAK1 = 1 and PAK3 = 1 and PDE10A = 1 and PFKM = 0 and PPFIBP1 = 1 and PRKCE = 0 and PRKG2 = 0 and PRPS1 = 0 and RAB3A = 0 and RASAL2 = 1 and RASGRP1 = 0 and REPS2 = 1 and SNAP25 = 1 and SOD1 = 1 and SV2B = 1 and SYT1 = 0 and TBC1D4 = 1 and TRO = 0 and TTC3 = 1 and TUBA4A = 0 and TUBB2A = 1 and UNC119 = 0 and VAV3 = 1 and VTI1B = 0 and YWHAH = 1
};

$Final1 :=
{ ADAM23 = 1 and AMPH = 0 and ARL3 = 1 and CACNA1C = 0 and CAMK2B = 1 and CYP2C8 = 0 and GAP43 = 0 and GFPT1 = 0 and GRIN1 = 1 and KCNA1 = 0 and KCNA2 = 1 and PDE10A = 0 and PPFIBP1 = 0 and PRKG2 = 0 and RAB3A = 1 and RASAL2 = 0 and REPS2 = 1 and SNAP25 = 0 and SOD1 = 0 and SYT1 = 1 and VAV3 = 0
};

$KnockDown2 :=
{ KO(SOD1) = 0 and KO(KCNA2) = 0 and KO(CACNA1C) = 0 and KO(SYT1) = 0 and KO(CDC42) = 1
};

$Initial2 :=
{ ADAM23 = 1 and AMPH = 1 and AP3B2 = 0 and ARHGAP44 = 0 and ARL3 = 0 and ATP1A3 = 1 and ATP6V1A = 1 and CACNA1C = 0 and CACNB2 = 0 and CACNB4 = 1 and CACNG2 = 1 and CADPS = 0 and CAMK2B = 1 and CCNB1 = 0 and CDC42 = 0 and CDK14 = 1 and CHML = 1 and CIT = 0 and CKS2 = 1 and CNTNAP1 = 0 and CYP2C8 = 1 and DNM1 = 1 and EEF1A2 = 0 and ENO2 = 1 and GAP43 = 0 and GFPT1 = 1 and GGH = 1 and GPI = 0 and GRIN1 = 1 and INA = 1 and KCNA1 = 0 and KCNA2 = 1 and KCNAB2 = 1 and MAP3K9 = 1 and MAPK8 = 1 and MCF2 = 0 and MYO1B = 1 and MYO5A = 0 and NEFH = 0 and NEFM = 0 and NELL2 = 1 and PAK1 = 1 and PAK3 = 1 and PDE10A = 1 and PFKM = 0 and PPFIBP1 = 1 and PRKCE = 0 and PRKG2 = 0 and PRPS1 = 0 and RAB3A = 0 and RASAL2 = 1 and RASGRP1 = 0 and REPS2 = 1 and SNAP25 = 1 and SOD1 = 1 and SV2B = 1 and SYT1 = 0 and TBC1D4 = 1 and TRO = 0 and TTC3 = 1 and TUBA4A = 0 and TUBB2A = 1 and UNC119 = 0 and VAV3 = 1 and VTI1B = 0 and YWHAH = 1
};

$Final2 :=
{ ADAM23 = 1 and AMPH = 0 and CACNB2 = 0 and CACNB4 = 0 and CACNG2 = 0 and CAMK2B = 1 and CHML = 1 and CYP2C8 = 0 and GFPT1 = 0 and GGH = 0 and GRIN1 = 1 and KCNA1 = 0 and KCNA2 = 0 and MAP3K9 = 1 and MYO5A = 1 and NELL2 = 1 and PDE10A = 0 and PPFIBP1 = 0 and RAB3A = 1 and RASGRP1 = 1 and SNAP25 = 0 and TBC1D4 = 1
};

$KnockDown3 :=
{ KO(SOD1) = 0 and KO(KCNA2) = 1 and KO(CACNA1C) = 0 and KO(SYT1) = 0 and KO(CDC42) = 0
};

$Initial3 :=
{ ADAM23 = 1 and AMPH = 1 and AP3B2 = 0 and ARHGAP44 = 0 and ARL3 = 0 and ATP1A3 = 1 and ATP6V1A = 1 and CACNA1C = 0 and CACNB2 = 0 and CACNB4 = 1 and CACNG2 = 1 and CADPS = 0 and CAMK2B = 1 and CCNB1 = 0 and CDC42 = 0 and CDK14 = 1 and CHML = 1 and CIT = 0 and CKS2 = 1 and CNTNAP1 = 0 and CYP2C8 = 1 and DNM1 = 1 and EEF1A2 = 0 and ENO2 = 1 and GAP43 = 0 and GFPT1 = 1 and GGH = 1 and GPI = 0 and GRIN1 = 1 and INA = 1 and KCNA1 = 0 and KCNA2 = 0 and KCNAB2 = 1 and MAP3K9 = 1 and MAPK8 = 1 and MCF2 = 0 and MYO1B = 1 and MYO5A = 0 and NEFH = 0 and NEFM = 0 and NELL2 = 1 and PAK1 = 1 and PAK3 = 1 and PDE10A = 1 and PFKM = 0 and PPFIBP1 = 1 and PRKCE = 0 and PRKG2 = 0 and PRPS1 = 0 and RAB3A = 0 and RASAL2 = 1 and RASGRP1 = 0 and REPS2 = 1 and SNAP25 = 1 and SOD1 = 1 and SV2B = 1 and SYT1 = 0 and TBC1D4 = 1 and TRO = 0 and TTC3 = 1 and TUBA4A = 0 and TUBB2A = 1 and UNC119 = 0 and VAV3 = 1 and VTI1B = 0 and YWHAH = 1
};

$Final3 :=
{ ADAM23 = 1 and AMPH = 0 and AP3B2 = 1 and ARHGAP44 = 0 and ARL3 = 1 and CACNA1C = 0 and CACNB4 = 0 and CAMK2B = 1 and CHML = 1 and DNM1 = 0 and GAP43 = 0 and GFPT1 = 0 and GRIN1 = 1 and KCNA1 = 0 and LIMK1 = 1 and MAP3K9 = 1 and MCF2 = 0 and MORF4L1 = 0 and MYO5A = 1 and NEFH = 1 and PDE10A = 0 and PPFIBP1 = 0 and PRKCE = 0 and PRKG2 = 0 and RAB3A = 1 and SH3GL2 = 0 and SNAP25 = 0 and SOD1 = 0 and STXBP1 = 1 and SYT1 = 1 and TTC3 = 0 and TUBB2A = 1 and VAV3 = 0 and YWHAH = 0
};

$KnockDown4 :=
{ KO(SOD1) = 1 and KO(KCNA2) = 0 and KO(CACNA1C) = 0 and KO(SYT1) = 0 and KO(CDC42) = 0
};

$Initial4 :=
{ ADAM23 = 1 and AMPH = 1 and AP3B2 = 0 and ARHGAP44 = 0 and ARL3 = 0 and ATP1A3 = 1 and ATP6V1A = 1 and CACNA1C = 0 and CACNB2 = 0 and CACNB4 = 1 and CACNG2 = 1 and CADPS = 0 and CAMK2B = 1 and CCNB1 = 0 and CDC42 = 0 and CDK14 = 1 and CHML = 1 and CIT = 0 and CKS2 = 1 and CNTNAP1 = 0 and CYP2C8 = 1 and DNM1 = 1 and EEF1A2 = 0 and ENO2 = 1 and GAP43 = 0 and GFPT1 = 1 and GGH = 1 and GPI = 0 and GRIN1 = 1 and INA = 1 and KCNA1 = 0 and KCNA2 = 1 and KCNAB2 = 1 and MAP3K9 = 1 and MAPK8 = 1 and MCF2 = 0 and MYO1B = 1 and MYO5A = 0 and NEFH = 0 and NEFM = 0 and NELL2 = 1 and PAK1 = 1 and PAK3 = 1 and PDE10A = 1 and PFKM = 0 and PPFIBP1 = 1 and PRKCE = 0 and PRKG2 = 0 and PRPS1 = 0 and RAB3A = 0 and RASAL2 = 1 and RASGRP1 = 0 and REPS2 = 1 and SNAP25 = 1 and SOD1 = 0 and SV2B = 1 and SYT1 = 0 and TBC1D4 = 1 and TRO = 0 and TTC3 = 1 and TUBA4A = 0 and TUBB2A = 1 and UNC119 = 0 and VAV3 = 1 and VTI1B = 0 and YWHAH = 1
};

$Final4 :=
{ ADAM23 = 1 and AMPH = 0 and APBA1 = 1 and ARHGAP44 = 0 and ARL3 = 1 and ATP1A3 = 1 and ATP6V1A = 0 and CACNA1C = 0 and CACNB2 = 1 and CACNB4 = 0 and CACNG2 = 0 and CADPS = 0 and CAMK2B = 1 and CNTNAP1 = 1 and CYP2C8 = 1 and DNM1 = 0 and EEF1A2 = 1 and ENO2 = 0 and GAP43 = 1 and GFPT1 = 0 and GGH = 0 and GRIN1 = 0 and KCNA1 = 0 and KCNA2 = 1 and LIMK1 = 1 and MAP3K9 = 1 and MAPK8 = 0 and MYO1B = 1 and MYO5A = 0 and NEFM = 1 and NELL2 = 0 and PAK1 = 0 and PAK3 = 0 and PDE10A = 0 and PFKM = 1 and PPFIBP1 = 0 and PRKG2 = 0 and RAB3A = 1 and RASAL2 = 0 and RASGRP1 = 1 and REPS2 = 1 and SH3GL2 = 0 and SNAP25 = 0 and SOD1 = 0 and STXBP1 = 1 and SV2B = 0 and TUBA4A = 1 and TUBB2A = 0 and VAV3 = 0 and VTI1B = 1 and YWHAH = 1
};

$KnockDown5 :=
{ KO(SOD1) = 0 and KO(KCNA2) = 0 and KO(CACNA1C) = 0 and KO(SYT1) = 1 and KO(CDC42) = 0
};

$Initial5 :=
{ ADAM23 = 1 and AMPH = 1 and AP3B2 = 0 and ARHGAP44 = 0 and ARL3 = 0 and ATP1A3 = 1 and ATP6V1A = 1 and CACNA1C = 0 and CACNB2 = 0 and CACNB4 = 1 and CACNG2 = 1 and CADPS = 0 and CAMK2B = 1 and CCNB1 = 0 and CDC42 = 0 and CDK14 = 1 and CHML = 1 and CIT = 0 and CKS2 = 1 and CNTNAP1 = 0 and CYP2C8 = 1 and DNM1 = 1 and EEF1A2 = 0 and ENO2 = 1 and GAP43 = 0 and GFPT1 = 1 and GGH = 1 and GPI = 0 and GRIN1 = 1 and INA = 1 and KCNA1 = 0 and KCNA2 = 1 and KCNAB2 = 1 and MAP3K9 = 1 and MAPK8 = 1 and MCF2 = 0 and MYO1B = 1 and MYO5A = 0 and NEFH = 0 and NEFM = 0 and NELL2 = 1 and PAK1 = 1 and PAK3 = 1 and PDE10A = 1 and PFKM = 0 and PPFIBP1 = 1 and PRKCE = 0 and PRKG2 = 0 and PRPS1 = 0 and RAB3A = 0 and RASAL2 = 1 and RASGRP1 = 0 and REPS2 = 1 and SNAP25 = 1 and SOD1 = 1 and SV2B = 1 and SYT1 = 0 and TBC1D4 = 1 and TRO = 0 and TTC3 = 1 and TUBA4A = 0 and TUBB2A = 1 and UNC119 = 0 and VAV3 = 1 and VTI1B = 0 and YWHAH = 1
};

$Final5 :=
{ AMPH = 0 and APBA1 = 1 and CACNA1C = 0 and CACNB4 = 0 and CACNG2 = 0 and CAMK2B = 1 and CHML = 1 and CYP2C8 = 0 and GAP43 = 0 and GFPT1 = 0 and GRIN1 = 1 and KCNA1 = 0 and LIMK1 = 1 and MORF4L1 = 0 and NELL2 = 1 and PDE10A = 0 and RASGRP1 = 1 and SNAP25 = 0 and SOD1 = 0 and SYT1 = 1 and TBC1D4 = 1 and VAV3 = 0
};
