import gzip
import json
from collections import defaultdict
from copy import deepcopy
from typing import Optional, List, Dict, Any

import biotite
import numpy as np
import torch
from Bio import Align
from Bio.pairwise2 import Alignment
from biotite.structure import AtomArray, filter_backbone, Atom
from biotite.structure import array as struct_array
from biotite.structure.io.pdb import PDBFile
from tqdm import tqdm

d3to1 = {
    'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'
}

d1to3 = {v: k for k, v in d3to1.items()}

d3to1_extended = {
    "A5N": "N", "A8E": "V", "A9D": "S", "AA3": "A", "AA4": "A", "AAR": "R",
    "ABA": "A", "ACL": "R", "AEA": "C", "AEI": "D", "AFA": "N", "AGM": "R",
    "AGQ": "Y", "AGT": "C", "AHB": "N", "AHL": "R", "AHO": "A", "AHP": "A",
    "AIB": "A", "AKL": "D", "AKZ": "D", "ALC": "A", "ALM": "A",
    "ALN": "A", "ALO": "T", "ALS": "A", "ALT": "A", "ALV": "A", "ALY": "K",
    "AME": "M", "AN6": "L", "AN8": "A", "API": "K", "APK": "K", "AR2": "R",
    "AR4": "E", "AR7": "R", "ARM": "R", "ARO": "R", "AS7": "N",
    "ASA": "D", "ASB": "D", "ASI": "D", "ASK": "D", "ASL": "D",
    "ASQ": "D", "AYA": "A", "AZH": "A", "AZK": "K", "AZS": "S",
    "AZY": "Y", "AVJ": "H", "A30": "Y", "A3U": "F", "ECC": "Q", "ECX": "C",
    "EFC": "C", "EHP": "F", "ELY": "K", "EME": "E", "EPM": "M", "EPQ": "Q",
    "ESB": "Y", "ESC": "M", "EXY": "L", "EXA": "K", "E0Y": "P", "E9V": "H",
    "E9M": "W", "EJA": "C", "EUP": "T", "EZY": "G", "E9C": "Y", "EW6": "S",
    "EXL": "W", "I2M": "I", "I4G": "G", "I58": "K", "IAM": "A", "IAR": "R",
    "ICY": "C", "IEL": "K", "IGL": "G", "IIL": "I", "ILG": "E",
    "ILM": "I", "ILX": "I", "ILY": "K", "IML": "I", "IOR": "R", "IPG": "G",
    "IT1": "K", "IYR": "Y", "IZO": "M", "IC0": "G", "M0H": "C", "M2L": "K",
    "M2S": "M", "M30": "G", "M3L": "K", "M3R": "K", "MA ": "A", "MAA": "A",
    "MAI": "R", "MBQ": "Y", "MC1": "S", "MCL": "K", "MCS": "C", "MD3": "C",
    "MD5": "C", "MD6": "G", "MDF": "Y", "ME0": "M", "MEA": "F", "MEG": "E",
    "MEN": "N", "MEQ": "Q", "MEU": "G", "MFN": "E", "MGG": "R",
    "MGN": "Q", "MGY": "G", "MH1": "H", "MH6": "S", "MHL": "L", "MHO": "M",
    "MHS": "H", "MHU": "F", "MIR": "S", "MIS": "S", "MK8": "L", "ML3": "K",
    "MLE": "L", "MLL": "L", "MLY": "K", "MLZ": "K", "MME": "M", "MMO": "R",
    "MNL": "L", "MNV": "V", "MP8": "P", "MPQ": "G", "MSA": "G", "MSE": "M",
    "MSL": "M", "MSO": "M", "MT2": "M", "MTY": "Y", "MVA": "V", "MYK": "K",
    "MYN": "R", "QCS": "C", "QIL": "I", "QMM": "Q", "QPA": "C", "QPH": "F",
    "Q3P": "K", "QVA": "C", "QX7": "A", "Q2E": "W", "Q75": "M", "Q78": "F",
    "QM8": "L", "QMB": "A", "QNQ": "C", "QNT": "C", "QNW": "C", "QO2": "C",
    "QO5": "C", "QO8": "C", "QQ8": "Q", "U2X": "Y", "U3X": "F", "UF0": "S",
    "UGY": "G", "UM1": "A", "UM2": "A", "UMA": "A", "UQK": "A", "UX8": "W",
    "UXQ": "F", "YCM": "C", "YOF": "Y", "YPR": "P", "YPZ": "Y", "YTH": "T",
    "Y1V": "L", "Y57": "K", "YHA": "K", "200": "F", "23F": "F", "23P": "A",
    "26B": "T", "28X": "T", "2AG": "A", "2CO": "C", "2FM": "M", "2GX": "F",
    "2HF": "H", "2JG": "S", "2KK": "K", "2KP": "K", "2LT": "Y", "2LU": "L",
    "2ML": "L", "2MR": "R", "2MT": "P", "2OR": "R", "2P0": "P", "2QZ": "T",
    "2R3": "Y", "2RA": "A", "2RX": "S", "2SO": "H", "2TY": "Y", "2VA": "V",
    "2XA": "C", "2ZC": "S", "6CL": "K", "6CW": "W", "6GL": "A", "6HN": "K",
    "60F": "C", "66D": "I", "6CV": "A", "6M6": "C", "6V1": "C", "6WK": "C",
    "6Y9": "P", "6DN": "K", "DA2": "R", "DAB": "A", "DAH": "F", "DBS": "S",
    "DBU": "T", "DBY": "Y", "DBZ": "A", "DC2": "C", "DDE": "H", "DDZ": "A",
    "DI7": "Y", "DHA": "S", "DHN": "V", "DIR": "R", "DLS": "K", "DM0": "K",
    "DMH": "N", "DMK": "D", "DNL": "K", "DNP": "A", "DNS": "K", "DNW": "A",
    "DOH": "D", "DON": "L", "DP1": "R", "DPL": "P", "DPP": "A", "DPQ": "Y",
    "DYS": "C", "D2T": "D", "DYA": "D", "DJD": "F", "DYJ": "P", "DV9": "E",
    "H14": "F", "H1D": "M", "H5M": "P", "HAC": "A", "HAR": "R", "HBN": "H",
    "HCM": "C", "HGY": "G", "HHI": "H", "HIA": "H", "HIP": "H",
    "HIQ": "H", "HL2": "L", "HLU": "L", "HMR": "R", "HNC": "C",
    "HOX": "F", "HPC": "F", "HPE": "F", "HPH": "F", "HPQ": "F", "HQA": "A",
    "HR7": "R", "HRG": "R", "HRP": "W", "HS8": "H", "HS9": "H", "HSE": "S",
    "HSK": "H", "HSL": "S", "HSO": "H", "HT7": "W", "HTI": "C", "HTR": "W",
    "HV5": "A", "HVA": "V", "HY3": "P", "HYI": "M", "HYP": "P", "HZP": "P",
    "HIX": "A", "HSV": "H", "HLY": "K", "HOO": "H", "H7V": "A", "L5P": "K",
    "LRK": "K", "L3O": "L", "LA2": "K", "LAA": "D", "LAL": "A", "LBY": "K",
    "LCK": "K", "LCX": "K", "LDH": "K", "LE1": "V", "LED": "L", "LEF": "L",
    "LEH": "L", "LEM": "L", "LEN": "L", "LET": "K", "LEX": "L",
    "LGY": "K", "LLO": "K", "LLP": "K", "LLY": "K", "LLZ": "K", "LME": "E",
    "LMF": "K", "LMQ": "Q", "LNE": "L", "LNM": "L", "LP6": "K", "LPD": "P",
    "LPG": "G", "LPS": "S", "LSO": "K", "LTR": "W", "LVG": "G", "LVN": "V",
    "LWY": "P", "LYF": "K", "LYK": "K", "LYM": "K", "LYN": "K", "LYO": "K",
    "LYP": "K", "LYR": "K", "LYU": "K", "LYX": "K", "LYZ": "K",
    "LAY": "L", "LWI": "F", "LBZ": "K", "P1L": "C", "P2Q": "Y", "P2Y": "P",
    "P3Q": "Y", "PAQ": "Y", "PAS": "D", "PAT": "W", "PBB": "C", "PBF": "F",
                "PCC": "P", "PCS": "F", "PE1": "K", "PEC": "C", "PF5": "F", #They had PCA mapped to Q, both are correct
    "PFF": "F", "PG1": "S", "PGY": "G", "PHA": "F", "PHD": "D",
    "PHI": "F", "PHL": "F", "PHM": "F", "PKR": "P", "PLJ": "P", "PM3": "F",
    "POM": "P", "PPN": "F", "PR3": "C", "PR4": "P", "PR7": "P", "PR9": "P",
    "PRJ": "P", "PRK": "K", "PRS": "P", "PRV": "G", "PSA": "F",
    "PSH": "H", "PTH": "Y", "PTM": "Y", "PTR": "Y", "PVH": "H", "PXU": "P",
    "PYA": "A", "PYH": "K", "PYX": "C", "PH6": "P", "P9S": "C", "P5U": "S",
    "POK": "R", "T0I": "Y", "T11": "F", "TAV": "D", "TBG": "V", "TBM": "T",
    "TCQ": "Y", "TCR": "W", "TEF": "F", "TFQ": "F", "TH5": "T", "TH6": "T",
    "THC": "T", "THZ": "R", "TIH": "A", "TIS": "S", "TLY": "K",
    "TMB": "T", "TMD": "T", "TNB": "C", "TNR": "S", "TNY": "T", "TOQ": "W",
    "TOX": "W", "TPJ": "P", "TPK": "P", "TPL": "W", "TPO": "T", "TPQ": "Y",
    "TQI": "W", "TQQ": "W", "TQZ": "C", "TRF": "W", "TRG": "K", "TRN": "W",
    "TRO": "W", "TRQ": "W", "TRW": "W", "TRX": "W", "TRY": "W",
    "TS9": "I", "TSY": "C", "TTQ": "W", "TTS": "Y", "TXY": "Y", "TY1": "Y",
    "TY2": "Y", "TY3": "Y", "TY5": "Y", "TY8": "Y", "TY9": "Y", "TYB": "Y",
    "TYC": "Y", "TYE": "Y", "TYI": "Y", "TYJ": "Y", "TYN": "Y", "TYO": "Y",
    "TYQ": "Y", "TYS": "Y", "TYT": "Y", "TYW": "Y", "TYY": "Y",
    "T8L": "T", "T9E": "T", "TNQ": "W", "TSQ": "F", "TGH": "W", "X2W": "E",
    "XCN": "C", "XPR": "P", "XSN": "N", "XW1": "A", "XX1": "K", "XYC": "A",
    "XA6": "F", "11Q": "P", "11W": "E", "12L": "P", "12X": "P", "12Y": "P",
    "143": "C", "1AC": "A", "1L1": "A", "1OP": "Y", "1PA": "F", "1PI": "A",
    "1TQ": "W", "1TY": "Y", "1X6": "S", "56A": "H", "5AB": "A", "5CS": "C",
    "5CW": "W", "5HP": "E", "5OH": "A", "5PG": "G", "51T": "Y", "54C": "W",
    "5CR": "F", "5CT": "K", "5FQ": "A", "5GM": "I", "5JP": "S", "5T3": "K",
    "5MW": "K", "5OW": "K", "5R5": "S", "5VV": "N", "5XU": "A", "55I": "F",
    "999": "D", "9DN": "N", "9NE": "E", "9NF": "F", "9NR": "R", "9NV": "V",
    "9E7": "K", "9KP": "K", "9WV": "A", "9TR": "K", "9TU": "K", "9TX": "K",
    "9U0": "K", "9IJ": "F", "B1F": "F", "B27": "T", "B2A": "A", "B2F": "F",
    "B2I": "I", "B2V": "V", "B3A": "A", "B3D": "D", "B3E": "E", "B3K": "K",
    "B3U": "H", "B3X": "N", "B3Y": "Y", "BB6": "C", "BB7": "C", "BB8": "F",
    "BB9": "C", "BBC": "C", "BCS": "C", "BCX": "C", "BFD": "D", "BG1": "S",
    "BH2": "D", "BHD": "D", "BIF": "F", "BIU": "I", "BL2": "L", "BLE": "L",
    "BLY": "K", "BMT": "T", "BNN": "F", "BOR": "R", "BP5": "A", "BPE": "C",
    "BSE": "S", "BTA": "L", "BTC": "C", "BTK": "K", "BTR": "W", "BUC": "C",
    "BUG": "V", "BYR": "Y", "BWV": "R", "BWB": "S", "BXT": "S", "F2F": "F",
    "F2Y": "Y", "FAK": "K", "FB5": "A", "FB6": "A", "FC0": "F", "FCL": "F",
    "FDL": "K", "FFM": "C", "FGL": "G", "FGP": "S", "FH7": "K", "FHL": "K",
    "FHO": "K", "FIO": "R", "FLA": "A", "FLE": "L", "FLT": "Y", "FME": "M",
    "FOE": "C", "FP9": "P", "FPK": "P", "FT6": "W", "FTR": "W", "FTY": "Y",
    "FVA": "V", "FZN": "K", "FY3": "Y", "F7W": "W", "FY2": "Y", "FQA": "K",
    "F7Q": "Y", "FF9": "K", "FL6": "D", "JJJ": "C", "JJK": "C", "JJL": "C",
    "JLP": "K", "J3D": "C", "J9Y": "R", "J8W": "S", "JKH": "P", "N10": "S",
    "N7P": "P", "NA8": "A", "NAL": "A", "NAM": "A", "NBQ": "Y", "NC1": "S",
    "NCB": "A", "NEM": "H", "NEP": "H", "NFA": "F", "NIY": "Y", "NLB": "L",
    "NLE": "L", "NLN": "L", "NLO": "L", "NLP": "L", "NLQ": "Q", "NLY": "G",
    "NMC": "G", "NMM": "R", "NNH": "R", "NOT": "L", "NPH": "C", "NPI": "A",
    "NTR": "Y", "NTY": "Y", "NVA": "V", "NWD": "A", "NYB": "C", "NYS": "C",
    "NZH": "H", "N80": "P", "NZC": "T", "NLW": "L", "N0A": "F", "N9P": "A",
    "N65": "K", "R1A": "C", "R4K": "W", "RE0": "W", "RE3": "W", "RGL": "R",
    "RGP": "E", "RT0": "P", "RVX": "S", "RZ4": "S", "RPI": "R", "RVJ": "A",
    "VAD": "V", "VAF": "V", "VAH": "V", "VAI": "V", "VB1": "K",
    "VH0": "P", "VR0": "R", "V44": "C", "V61": "F", "VPV": "K", "V5N": "H",
    "V7T": "K", "Z01": "A", "Z3E": "T", "Z70": "H", "ZBZ": "C", "ZCL": "F",
    "ZU0": "T", "ZYJ": "P", "ZYK": "P", "ZZD": "C", "ZZJ": "A", "ZIQ": "W",
    "ZPO": "P", "ZDJ": "Y", "ZT1": "K", "30V": "C", "31Q": "C", "33S": "F",
    "33W": "A", "34E": "V", "3AH": "H", "3BY": "P", "3CF": "F", "3CT": "Y",
    "3GA": "A", "3GL": "E", "3MD": "D", "3MY": "Y", "3NF": "Y", "3O3": "E",
    "3PX": "P", "3QN": "K", "3TT": "P", "3XH": "G", "3YM": "Y", "3WS": "A",
    "3WX": "P", "3X9": "C", "3ZH": "H", "7JA": "I", "73C": "S", "73N": "R",
    "73O": "Y", "73P": "K", "74P": "K", "7N8": "F", "7O5": "A", "7XC": "F",
    "7ID": "D", "7OZ": "A", "C1S": "C", "C1T": "C", "C1X": "K", "C22": "A",
    "C3Y": "C", "C4R": "C", "C5C": "C", "C6C": "C", "CAF": "C", "CAS": "C",
    "CAY": "C", "CCS": "C", "CEA": "C", "CGA": "E", "CGU": "E", "CGV": "C",
    "CHP": "G", "CIR": "R", "CLE": "L", "CLG": "K", "CLH": "K", "CME": "C",
    "CMH": "C", "CML": "C", "CMT": "C", "CR5": "G", "CS0": "C", "CS1": "C",
    "CS3": "C", "CS4": "C", "CSA": "C", "CSB": "C", "CSD": "C", "CSE": "C",
    "CSJ": "C", "CSO": "C", "CSP": "C", "CSR": "C", "CSS": "C", "CSU": "C",
    "CSW": "C", "CSX": "C", "CSZ": "C", "CTE": "W", "CTH": "T", "CWD": "A",
    "CWR": "S", "CXM": "M", "CY0": "C", "CY1": "C", "CY3": "C", "CY4": "C",
    "CYA": "C", "CYD": "C", "CYF": "C", "CYG": "C", "CYJ": "K", "CYM": "C",
    "CYQ": "C", "CYR": "C", "CYW": "C", "CZ2": "C", "CZZ": "C",
    "CG6": "C", "C1J": "R", "C4G": "R", "C67": "R", "C6D": "R", "CE7": "N",
    "CZS": "A", "G01": "E", "G8M": "E", "GAU": "E", "GEE": "G", "GFT": "S",
    "GHC": "E", "GHG": "Q", "GHW": "E", "GL3": "G", "GLH": "Q", "GLJ": "E",
    "GLK": "E", "GLQ": "E", "GLZ": "G",
    "GMA": "E", "GME": "E", "GNC": "Q", "GPL": "K", "GSC": "G", "GSU": "E",
    "GT9": "C", "GVL": "S", "G3M": "R", "G5G": "L", "G1X": "Y", "G8X": "P",
    "K1R": "C", "KBE": "K", "KCX": "K", "KFP": "K", "KGC": "K", "KNB": "A",
    "KOR": "M", "KPI": "K", "KPY": "K", "KST": "K", "KYN": "W", "KYQ": "K",
    "KCR": "K", "KPF": "K", "K5L": "S", "KEO": "K", "KHB": "K", "KKD": "D",
    "K5H": "C", "K7K": "S", "OAR": "R", "OAS": "S", "OBS": "K", "OCS": "C",
    "OCY": "C", "OHI": "H", "OHS": "D", "OLD": "H", "OLT": "T", "OLZ": "S",
    "OMH": "S", "OMT": "M", "OMX": "Y", "OMY": "Y", "ONH": "A", "ORN": "A",
    "ORQ": "R", "OSE": "S", "OTH": "T", "OXX": "D", "OYL": "H", "O7A": "T",
    "O7D": "W", "O7G": "V", "O2E": "S", "O6H": "W", "OZW": "F", "S12": "S",
    "S1H": "S", "S2C": "C", "S2P": "A", "SAC": "S", "SAH": "C", "SAR": "G",
    "SBG": "S", "SBL": "S", "SCH": "C", "SCS": "C", "SCY": "C", "SD4": "N",
    "SDB": "S", "SDP": "S", "SEB": "S", "SEE": "S", "SEG": "A", "SEL": "S",
    "SEM": "S", "SEN": "S", "SEP": "S", "SET": "S", "SGB": "S",
    "SHC": "C", "SHP": "G", "SHR": "K", "SIB": "C", "SLL": "K", "SLZ": "K",
    "SMC": "C", "SME": "M", "SMF": "F", "SNC": "C", "SNN": "N", "SOY": "S",
    "SRZ": "S", "STY": "Y", "SUN": "S", "SVA": "S", "SVV": "S", "SVW": "S",
    "SVX": "S", "SVY": "S", "SVZ": "S", "SXE": "S", "SKH": "K", "SNM": "S",
    "SNK": "H", "SWW": "S", "WFP": "F", "WLU": "L", "WPA": "F", "WRP": "W",
    "WVL": "V", "02K": "A", "02L": "N", "02O": "A", "02Y": "A", "033": "V",
    "037": "P", "03Y": "C", "04U": "P", "04V": "P", "05N": "P", "07O": "C",
    "0A0": "D", "0A1": "Y", "0A2": "K", "0A8": "C", "0A9": "F", "0AA": "V",
    "0AB": "V", "0AC": "G", "0AF": "W", "0AG": "L", "0AH": "S", "0AK": "D",
    "0AR": "R", "0BN": "F", "0CS": "A", "0E5": "T", "0EA": "Y", "0FL": "A",
    "0LF": "P", "0NC": "A", "0PR": "Y", "0QL": "C", "0TD": "D", "0UO": "W",
    "0WZ": "Y", "0X9": "R", "0Y8": "P", "4AF": "F", "4AR": "R", "4AW": "W",
    "4BF": "F", "4CF": "F", "4CY": "M", "4DP": "W", "4FB": "P", "4FW": "W",
    "4HL": "Y", "4HT": "W", "4IN": "W", "4MM": "M", "4PH": "F", "4U7": "A",
    "41H": "F", "41Q": "N", "42Y": "S", "432": "S", "45F": "P", "4AK": "K",
    "4D4": "R", "4GJ": "C", "4KY": "P", "4L0": "P", "4LZ": "Y", "4N7": "P",
    "4N8": "P", "4N9": "P", "4OG": "W", "4OU": "F", "4OV": "S", "4OZ": "S",
    "4PQ": "W", "4SJ": "F", "4WQ": "A", "4HH": "S", "4HJ": "S", "4J4": "C",
    "4J5": "R", "4II": "F", "4VI": "R", "823": "N", "8SP": "S", "8AY": "A",
}

d3to1_all = {**d3to1, **d3to1_extended}

n_ca_dist = 1.46
ca_c_dist = 1.54
c_n_dist = 1.34

MissingResidues = List[int]


def order_keys_in_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: d[k]
        for k in sorted(d.keys())
    }


def get_pdb_file_pointer(file_name) -> PDBFile:
    opener = gzip.open if file_name.endswith(".gz") else open
    with opener(str(file_name), "rt") as f:
        source = PDBFile.read(f)

    return source


def read_pdb_file(file_name: str) -> Optional[AtomArray]:
    source = get_pdb_file_pointer(file_name)
    if source.get_model_count() > 1:
        return None
    # Pull out the atomarray from atomarraystack
    source_struct = source.get_structure()[0]

    return source_struct


def filter_only_ca(structure: AtomArray) -> AtomArray:
    return structure[structure.atom_name == "CA"]


def get_chains(structure: AtomArray) -> List[str]:
    return list(set(structure.chain_id))


def split_structure_by_chains(structure: AtomArray) -> Dict[str, AtomArray]:
    chains = get_chains(structure)
    return {
        chain: structure[structure.chain_id == chain]
        for chain in chains
    }


def get_num_residues_id(structure: AtomArray) -> int:
    return int(max(set(structure.res_id)))


def get_lowest_residue_id(structure: AtomArray) -> int:
    # In case of missing residues, the lowest residue is not 1
    # Also in case there are residues indexed with negative numbers
    # or with 0
    return int(min(set(structure.res_id)))


def iter_residue_ids(structure: AtomArray) -> range:
    return range(get_lowest_residue_id(structure), get_num_residues_id(structure) + 1)

def get_num_residues_by_chains(structure: AtomArray) -> Dict[str, int]:
    chains = get_chains(structure)
    return {
        chain: get_num_residues_id(structure[structure.chain_id == chain])
        for chain in chains
    }


def get_num_residues_backbone_by_chains(structure: AtomArray) -> Dict[str, int]:
    chains = get_chains(structure)
    structure = filter_only_ca(structure)
    return {
        chain: get_num_residues_id(structure[structure.chain_id == chain])
        for chain in chains
    }


def read_pdb_header(file_name: str) -> List[str]:
    source = get_pdb_file_pointer(file_name)
    body_lines = ["ATOM", "HETATM"]
    stop_pointer = _search_body_lines(body_lines, source)

    return source.lines[:stop_pointer]


def read_pdb_body(file_name: str) -> List[str]:
    source = get_pdb_file_pointer(file_name)
    body_lines = ["ATOM", "HETATM"]
    start_pointer = _search_body_lines(body_lines, source)

    return source.lines[start_pointer:]


def _search_body_lines(body_lines, source):
    for i, line in enumerate(source.lines):
        for body_line in body_lines:
            if line.startswith(body_line):
                return i
    return 0


def get_sequence_from_header(header: List[str]) -> Optional[Dict[str, str]]:
    seqres_lines = [
        line for line in header
        if line.startswith("SEQRES")
    ]

    if len(seqres_lines) == 0:
        return None

    # A seqres example
    # 'SEQRES   4 D  109  ALA ASN TYR CYS SER GLY GLU CYS GLU PHE VAL PHE LEU          '

    # Split the lines in columns
    seq_res_columns = [line.split() for line in seqres_lines]

    # Get the sequence from each line
    current_chain = None
    amino_acids_by_chain = defaultdict(list)
    for line in seq_res_columns:
        for i, column in enumerate(line):
            if i == 0 or i == 1 or i == 3:
                continue

            if i == 2:
                current_chain = column
                continue

            amino_acid = d3to1_all.get(column, "?")
            amino_acids_by_chain[current_chain].append(amino_acid)

    # Join the sequences
    amino_acids_by_chain = {
        chain: "".join(seq)
        for chain, seq in amino_acids_by_chain.items()
    }

    return amino_acids_by_chain


def get_sequence_from_structure(structure: AtomArray) -> Dict[str, str]:
    chains = get_chains(structure)
    structure = filter_only_ca(structure)
    return {
        chain: "".join(
            d3to1_all.get(res_name, "?")
            for res_name in structure[structure.chain_id == chain].res_name
        )
        for chain in chains
    }


def sequence_alignments(header_seq: str, structure_seq: str) -> Alignment:
    aligner = Align.PairwiseAligner()
    aligner.gap_score = -1
    aligner.mismatch_score = -1
    aligner.extend_gap_score = -0.5

    alignments = aligner.align(header_seq, structure_seq)

    return alignments


def missing_residues_by_sequence_alignment(
        header_seq: str,
        structure_seq: str,
        all_alignments: bool = False
) -> List[MissingResidues]:

    alignments = sequence_alignments(header_seq, structure_seq)

    missing_residues_by_alignment = []
    for alignment in alignments:
        missing_residues = missing_residue_in_alignment(alignment)
        missing_residues_by_alignment.append(missing_residues)

        if not all_alignments:
            break

    return missing_residues_by_alignment


def missing_residue_in_alignment(alignment: Alignment) -> MissingResidues:
    missing_residues = []
    for i, (header_res, structure_res) in enumerate(zip(alignment[0], alignment[1])):
        if header_res != structure_res:
            missing_residues.append(i + 1)

    return missing_residues


def missing_residues_by_sequence_alignment_by_chains(
        header_seq: Dict[str, str],
        structure_seq: Dict[str, str],
        all_alignments: bool = False
) -> Dict[str, List[MissingResidues]]:
    missing_residues_by_chains = {}
    for chain in header_seq.keys():
        missing_residues_by_chains[chain] = missing_residues_by_sequence_alignment(
            header_seq[chain], structure_seq[chain], all_alignments=all_alignments
        )

    return missing_residues_by_chains


def missing_residues_by_structure_look_residues_id(structure: AtomArray) -> Dict[str, MissingResidues]:
    chains = get_chains(structure)
    structure = filter_only_ca(structure)
    missing_residues_by_chains = {}
    for chain in chains:
        missing_residues_by_chains[chain] = []
        chain_structure = structure[structure.chain_id == chain]

        for i in iter_residue_ids(chain_structure):
            if len(chain_structure[chain_structure.res_id == i]) == 0:
                missing_residues_by_chains[chain].append(i)

    return missing_residues_by_chains


def missing_residues_by_structure_continuity(
        structure: AtomArray,
        broken_only: bool = False
) -> Dict[str, MissingResidues]:
    chains = get_chains(structure)
    missing_residues_by_chains = {}
    for chain in chains:
        chain_structure = structure[(structure.chain_id == chain)]
        stop_indexes_atoms: np.ndarray = biotite.structure.check_bond_continuity(chain_structure)

        start_indexes_residues = chain_structure[stop_indexes_atoms-1].res_id
        stop_indexes_residues = chain_structure[stop_indexes_atoms].res_id

        missing_residues_by_chains[chain] = []
        for start, stop in zip(start_indexes_residues, stop_indexes_residues):
            in_between = list(range(start+1, stop))
            broken = False
            if not len(in_between):
                broken = True
                in_between = [start, stop]

            if broken_only and not broken:
                continue

            missing_residues_by_chains[chain].extend(in_between)

        missing_residues_by_chains[chain] = sorted(list(set(missing_residues_by_chains[chain])))

    return missing_residues_by_chains


def broken_residues_by_structure(structure: AtomArray):
    backbone_mask = filter_backbone(structure)
    backbone_structure = structure[backbone_mask]
    chains = get_chains(structure)

    broken_residues_by_chains = {}
    for chain in chains:
        broken_residues_by_chains[chain] = []

        chain_structure = backbone_structure[backbone_structure.chain_id == chain]

        for i in iter_residue_ids(chain_structure):
            backbone_residue = chain_structure[chain_structure.res_id == i]
            n_atom = backbone_residue[backbone_residue.atom_name == "N"]
            ca_atom = backbone_residue[backbone_residue.atom_name == "CA"]
            c_atom = backbone_residue[backbone_residue.atom_name == "C"]

            if len(backbone_residue) == 0:
                continue

            if len(n_atom) == 0 or len(ca_atom) == 0 or len(c_atom) == 0:
                broken_residues_by_chains[chain].append(i)
                continue

            if n_atom.res_id[0] > ca_atom.res_id[0] or ca_atom.res_id[0] > c_atom.res_id[0]:
                broken_residues_by_chains[chain].append(i)
                continue

    return broken_residues_by_chains


def determine_missing_residues(pdb_file: str) -> Dict[str, MissingResidues]:

    # Get the info
    header = read_pdb_header(pdb_file)
    structure = read_pdb_file(pdb_file)

    # Compute all the missing residues possibilities
    header_seq = get_sequence_from_header(header)
    missing_residues_header = None
    if header_seq is not None:
        # If the sequence is not in the header, we can't compute the missing residues with this method
        structure_seq = get_sequence_from_structure(structure)
        missing_residues_header = missing_residues_by_sequence_alignment_by_chains(
            header_seq, structure_seq, all_alignments=True
        )

    missing_residues_struct_res = missing_residues_by_structure_look_residues_id(structure)
    missing_residues_struct_cont = missing_residues_by_structure_continuity(structure)
    broken_residues_struct = broken_residues_by_structure(structure)

    missing_residues = missing_residues_struct_res
    if missing_residues_header is not None:

        # Reindex the missing residues by header
        structure = reindex_alignments(missing_residues_header, structure)

        # We need to drop all the alignments which include already existent residues
        missing_residues_header = keep_only_plausible_alignments(missing_residues_header, structure)

        # We need to check all the given alignments to decide which is the best for each chain
        for chain in missing_residues_header.keys():
            alignment_found = False
            for alignment in missing_residues_header[chain]:
                if set(missing_residues_struct_res[chain]).issubset(set(alignment)):
                    continue

                # If the alignment is correct, we can use it
                missing_residues[chain] = alignment
                alignment_found = True
                break

            # If we didn't find any alignment, we will use the first alignment from the header
            # which also has the highest score
            if not alignment_found and len(missing_residues_header[chain]) > 0:
                missing_residues[chain] = missing_residues_header[chain][0]

    # And now we augment the missing residues with the broken residues
    for chain in broken_residues_struct.keys():
        missing_residues[chain].extend(broken_residues_struct[chain])
        missing_residues[chain] = sorted(list(set(missing_residues[chain])))

    # And we also augment with the continuity missing residues
    for chain in missing_residues_struct_cont.keys():
        missing_residues[chain].extend(missing_residues_struct_cont[chain])
        missing_residues[chain] = sorted(list(set(missing_residues[chain])))

    # Cast the elements of the list to int
    for chain in missing_residues.keys():
        missing_residues[chain] = [int(x) for x in missing_residues[chain]]

    return missing_residues


def reindex_alignments(
        missing_residues_header: Dict[str, List[MissingResidues]],
        structure: AtomArray
):
    for chain in missing_residues_header.keys():
        chain_struct = structure[structure.chain_id == chain]
        start_index = get_lowest_residue_id(chain_struct)

        for i, alignment in enumerate(missing_residues_header[chain]):
            indexes = np.array(alignment) + start_index - 1
            missing_residues_header[chain][i] = indexes.tolist()

    return structure


def keep_only_plausible_alignments(missing_residues_header, structure):
    missing_residues_plausible = defaultdict(list)
    for chain in missing_residues_header.keys():
        for i, alignment in enumerate(missing_residues_header[chain]):
            # check if all the residues in alignment exist in the structure
            structure_chain = structure[structure.chain_id == chain]
            res_ids_in_structure = set(structure_chain.res_id)

            # Check if any of the residues in the alignment is in the structure
            if len(set(alignment).intersection(res_ids_in_structure)) > 0:
                continue

            # If not, we can use this alignment
            missing_residues_plausible[chain].append(alignment)
    missing_residues_header = missing_residues_plausible
    return missing_residues_header


def get_all_residues_id(
        structure: AtomArray,
        missing_residues_id: Dict[str, MissingResidues]
) -> Dict[str, List[int]]:

    all_residues_id = {
        chain: []
        for chain in missing_residues_id.keys()
    }
    for chain in missing_residues_id.keys():
        chain_struct = structure[structure.chain_id == chain]
        chain_residues = set(missing_residues_id[chain]).union(set(chain_struct.res_id))
        all_residues_id[chain] = sorted(list(chain_residues))

    return all_residues_id

from biotite.sequence import ProteinSequence
from biotite.sequence.align import align_optimal, SubstitutionMatrix
from biotite.structure.io.pdb import PDBFile

def check_working_alignment(arr):
    # find the first and last non-missing value in the array
    start, end = None, None
    for i in range(len(arr)):
        if arr[i] != -1:
            start = i
            break
    for i in range(len(arr)-1, -1, -1):
        if arr[i] != -1:
            end = i
            break
    
    # if there are no non-missing values, the array is monotonic by default
    if start is None or end is None:
        return True
    
    # check that the missing values can be replaced with consecutive numbers
    prev_val = arr[start]
    num_consecutive_missing = 0
    for i in range(start+1, end+1):
        if arr[i] == -1:
            num_consecutive_missing += 1
        else:
            # fill in the missing values with consecutive numbers
            if num_consecutive_missing > 0:
                if num_consecutive_missing > arr[i] - prev_val - 1:
                    return False
                for j in range(num_consecutive_missing):
                    arr[i-j-1] = arr[i] - j - 1
                num_consecutive_missing = 0
            # check that the array is still monotonic
            if arr[i] < prev_val:
                return False
            prev_val = arr[i]
    
    # fill in any missing values at the end of the array
    if num_consecutive_missing > 0:
        if num_consecutive_missing > arr[end] - prev_val:
            return False
        for j in range(num_consecutive_missing):
            arr[end-j] = prev_val + j + 1
    
    return True


def replace_monotonic(arr):
    # Find the first non-negative value in the array
    i = 0
    while i < len(arr) and arr[i] == -1:
        i += 1
    
    # If the first value is -1, backfill with consecutive integers starting from i
    if i > 0:
        for j in range(i-1,-1,-1):
            arr[j] = arr[j + 1] - 1
    
    prev = arr[i]    
    # Iterate over array starting from the first non-negative value
    for j in range(i+1, len(arr)):
        # If the element is -1, replace with consecutive integer
        if arr[j] == -1:
            arr[j] = prev + 1
            
            # Ensure that the array stays monotonic
            if arr[j] <= arr[j-1]:
                arr[j] = arr[j-1] + 1
                
            # Update variables
            prev = arr[j]
                
        # If the element is not -1, update variables
        else:
            prev = arr[j]
    
    # Return modified array
    return arr


def align_structure_to_sequence(seq: str, pdb_file: str):
    # Read structure and get sequence from it
    struct = PDBFile.read(pdb_file).get_structure(model=1)
    struct_res = [struct[struct.res_id == r].res_name[0] for r in np.unique(struct.res_id)]
    extracted_seq_from_struct = "".join(d3to1_all.get(res_name, '*') for res_name in struct_res)
    seq = seq.replace('?', '*')

    # Align given sequence to structure to find gaps
    alignment = align_optimal(
        ProteinSequence(seq),
        ProteinSequence(extracted_seq_from_struct),
        SubstitutionMatrix.std_protein_matrix()
    )

    # Find alignment that makes sense with structure's residue ids
    seq_res_map = {i: n for i, n in enumerate(np.unique(struct.res_id))}
    residue_ids = []
    for aln in alignment:
        residue_ids = [seq_res_map.get(i, -1) for i in aln.trace[:,-1]]
        if check_working_alignment(residue_ids):
            break

    if residue_ids == []:
        raise Exception("Structure does not align to sequence")
    
    # Return residue ids and names of gaps to fill
    residue_ids = replace_monotonic(residue_ids)
    seqs = aln.get_gapped_sequences()
    gaps = [i for i,c in enumerate(seqs[1]) if c == '-']
    residues = [seqs[0][i] for i in gaps]
    res_id = [residue_ids[i] for i in gaps]
    return list(zip(res_id, residues))


def mock_missing_info(
        in_pdb_file: str,
        out_pdb_file: str
):
    structure = read_pdb_file(in_pdb_file)
    missing_residues_id = determine_missing_residues(in_pdb_file)
    all_residues_id = get_all_residues_id(structure, missing_residues_id)

    header = read_pdb_header(in_pdb_file)
    header_seq = get_sequence_from_header(header)
    chains = get_chains(structure)

    new_residues = {
        chain: []
        for chain in chains
    }

    for chain in chains:
        chain_structure = structure[structure.chain_id == chain]
        start_index = get_lowest_residue_id(chain_structure)

        for residue_id in all_residues_id[chain]:

            if residue_id not in missing_residues_id[chain]:
                residue = chain_structure[chain_structure.res_id == residue_id]
                new_residues[chain].append(residue)
                continue

            missing_aa_3 = "GLY"
            # This is the case for broken existent residues
            if residue_id in chain_structure.res_id:
                first_atom = chain_structure[chain_structure.res_id == residue_id][0]
                missing_aa_3 = first_atom.res_name

            # This is the case for missing residues
            elif header_seq is not None:
                index = residue_id - start_index
                missing_aa = header_seq[chain][index]
                missing_aa_3 = d1to3[missing_aa]

            n_atom = Atom(
                coord=np.random.rand(3),
                chain_id=chain,
                res_id=residue_id,
                res_name=missing_aa_3,
                atom_name="N",
                element="N",
            )

            ca_atom = Atom(
                coord=np.random.rand(3),
                chain_id=chain,
                res_id=residue_id,
                res_name=missing_aa_3,
                atom_name="CA",
                element="C",
            )

            c_atom = Atom(
                coord=np.random.rand(3),
                chain_id=chain,
                res_id=residue_id,
                res_name=missing_aa_3,
                atom_name="C",
                element="C",
            )

            residue = [n_atom, ca_atom, c_atom]
            new_residue = struct_array(residue)
            new_residues[chain].append(new_residue)

    atom_arrays = []
    for chain in chains:
        for residue in new_residues[chain]:
            atom_arrays.extend(residue)

    new_atom_array = struct_array(atom_arrays)
    write_structure_to_pdb(new_atom_array, out_pdb_file)

    # add the old header to the new file
    add_header_info(header, out_pdb_file)

    start_indexes = {
        chain: get_lowest_residue_id(structure[structure.chain_id == chain])
        for chain in chains
    }

    missing_info = {
        "missing_residues_id": missing_residues_id,
        "start_indexes": start_indexes
    }

    # write in out_pdb_file + ".missing" the missing residues as a json
    with open(out_pdb_file + ".missing", "w") as f:
        json.dump(missing_info, f)

    return new_atom_array


def add_header_info(
        header: List[str],
        out_pdb_file: str
):
    body = read_pdb_body(out_pdb_file)

    new_pdb_lines = header + body
    for i, line in enumerate(new_pdb_lines):
        if not line.endswith("\n"):
            new_pdb_lines[i] = line + "\n"

    with open(out_pdb_file, "w") as f:
        f.write("".join(new_pdb_lines))


def determine_quality_of_structure(
        structure: AtomArray,
):
    """
    This is a very simple function that determines the quality of a structure based
    on the distances between the backbone atoms. Because of that, it only works
    for fully defined structures.

    The lower the score, the better the structure.

    Quality range: [0, inf)
    """

    chains = get_chains(structure)
    quality = {
        chain: 0.0
        for chain in chains
    }

    for chain in chains:
        chain_structure = structure[structure.chain_id == chain]

        # filter just the backbone of the chain
        backbone = chain_structure[
            filter_backbone(chain_structure)
        ]

        # get the coordinates of the backbone
        backbone_coords = backbone.coord

        # get the distances between the backbone atoms
        atom_dist = np.linalg.norm(backbone_coords[1:] - backbone_coords[:-1], axis=1)

        n_ca_len_diff = np.abs(atom_dist[::3] - n_ca_dist)
        ca_c_len_diff = np.abs(atom_dist[1::3] - ca_c_dist)
        c_n_len_diff = np.abs(atom_dist[2::3] - c_n_dist)

        # the quality of a chain is the sum of the differences
        quality[chain] = np.sum(
            np.concatenate([n_ca_len_diff, ca_c_len_diff, c_n_len_diff])
        )

    # Quality score is the mean of the quality of each chain
    quality_score = np.mean(list(quality.values()))

    return quality_score


def compute_backbone_distances(structure: AtomArray):
    chains = get_chains(structure)

    backbone_distances = {
        chain: {
            "n_ca": [],
            "ca_c": [],
            "c_n": [],
        }
        for chain in chains
    }
    for chain in chains:
        chain_structure = structure[structure.chain_id == chain]

        # filter just the backbone of the chain
        backbone = chain_structure[
            filter_backbone(chain_structure)
        ]

        # get the coordinates of the backbone
        backbone_coords = backbone.coord

        # get the distances between the backbone atoms
        atom_dist = np.linalg.norm(backbone_coords[1:] - backbone_coords[:-1], axis=1)

        n_ca_dist = atom_dist[::3]
        ca_c_dist = atom_dist[1::3]
        c_n_dist = atom_dist[2::3]

        backbone_distances[chain]["n_ca"] = n_ca_dist
        backbone_distances[chain]["ca_c"] = ca_c_dist
        backbone_distances[chain]["c_n"] = c_n_dist

    return backbone_distances


def compute_median_backbone_distances(structure: AtomArray):
    backbone_distances = compute_backbone_distances(structure)

    median_backbone_distances = {
        "n_ca": np.median(np.concatenate([backbone_distances[chain]["n_ca"] for chain in backbone_distances])),
        "ca_c": np.median(np.concatenate([backbone_distances[chain]["ca_c"] for chain in backbone_distances])),
        "c_n": np.median(np.concatenate([backbone_distances[chain]["c_n"] for chain in backbone_distances])),
    }

    return median_backbone_distances


def backbone_loss_function(coords: torch.Tensor, pad_mask: torch.Tensor):
    # coords.shape = [batch_size, num_atoms, 3]
    # pad_mask.shape = [batch_size, num_atoms]

    # create masks
    atom_dist_pad_mask = pad_mask[:, 1:]
    n_ca_dist_pad_mask = atom_dist_pad_mask[:, 0::3]
    ca_c_dist_pad_mask = atom_dist_pad_mask[:, 1::3]
    c_n_dist_pad_mask = atom_dist_pad_mask[:, 2::3]

    # compute distances
    atom_dist = torch.norm(coords[:, 1:] - coords[:, :-1], dim=2)
    n_ca_dist_diff = (atom_dist[:, 0::3] - n_ca_dist) ** 2
    ca_c_dist_diff = (atom_dist[:, 1::3] - ca_c_dist) ** 2
    c_n_dist_diff = (atom_dist[:, 2::3] - c_n_dist) ** 2

    # apply mask
    n_ca_dist_diff[~n_ca_dist_pad_mask] = 0
    ca_c_dist_diff[~ca_c_dist_pad_mask] = 0
    c_n_dist_diff[~c_n_dist_pad_mask] = 0

    all_diffs = torch.cat([n_ca_dist_diff, ca_c_dist_diff, c_n_dist_diff], dim=1)
    return all_diffs.sum()


def write_structure_to_pdb(structure: AtomArray, file_name: str) -> str:
    sink = PDBFile()
    sink.set_structure(structure)
    sink.write(file_name)

    return file_name


def atom_contact_loss_function(
        coords: torch.Tensor,
        pad_mask: torch.Tensor,
        contact_distance=1.2
):
    # coords shape = [batch_size, num_atoms, 3]
    # pad_mask shape = [batch_size, num_atoms]

    # Make a distance matrix
    dist_mat = torch.cdist(coords, coords)

    # Extend the pad_mask to dist_mat shape
    pad_mask_matrix = pad_mask.unsqueeze(1) * pad_mask.unsqueeze(2)

    # Compute the diagonal and non-contact masks
    diagonal_mask = torch.eye(dist_mat.shape[1], device=coords.device, dtype=torch.bool)
    non_contacts = dist_mat > contact_distance

    # Inverse the distance matrix
    dist_mat = contact_distance - dist_mat
    dist_mat[non_contacts] = 0
    dist_mat[:, diagonal_mask] = 0
    dist_mat[~pad_mask_matrix] = 0

    return dist_mat.sum()


def gradient_descent_on_physical_constraints(
        coords: torch.Tensor,
        inpaint_mask: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        num_epochs: int = 1000,
        device: torch.device = torch.device("cpu"),
        stop_patience: int = 10,
        show_progress: bool = True,
        lr: float = 0.01,
):
    coords = torch.nn.Parameter(coords.to(device))
    optimizer = torch.optim.Adam([coords], lr=lr)
    inpaint_mask = inpaint_mask.to(device)

    if pad_mask is None:
        pad_mask = torch.ones_like(inpaint_mask)
    pad_mask = pad_mask.to(device)

    constant_loss = 0
    prev_loss = 0.0

    progress_bar = tqdm(range(num_epochs), disable=not show_progress)
    for _ in progress_bar:
        optimizer.zero_grad()
        loss1 = backbone_loss_function(coords, pad_mask)
        loss2 = atom_contact_loss_function(coords, pad_mask)

        loss = loss1 + loss2
        loss_value = loss.detach().cpu().item()

        progress_bar.set_description(f"Fine tuning - Loss: {loss_value:.3f}")

        # We want to apply the backprop only on the masked values
        loss.backward(retain_graph=True)
        coords._grad[~inpaint_mask] = 0
        optimizer.step()

        if abs(loss_value - prev_loss) < 1e-3:
            constant_loss += 1
        else:
            constant_loss = 0

        if constant_loss >= stop_patience:
            break

        prev_loss = loss_value

    return coords.detach()


def add_sequence_to_pdb_header(
        in_pdb_file: str,
        sequence: str,
        chain: Optional[str] = None,
        out_pdb_file: Optional[str] = None
):
    header = read_pdb_header(in_pdb_file)
    body = read_pdb_body(in_pdb_file)

    if chain is None:
        structure = read_pdb_file(in_pdb_file)
        chain = get_chains(structure)[0]

    # a line in the header that has sequence should look like this
    # Example:
    # SEQRES   4 D  109  ALA ASN TYR CYS SER GLY GLU CYS GLU PHE VAL PHE LEU
    # General:
    # SEQRES   <line> <chain> <num_residues> <residues> <max 13 residues per line>

    # More rules for the header seqres
    """
     1 -  6        Record name    "SEQRES"
     8 - 10        Integer        serNum       Serial number of the SEQRES record for  the
                                               current  chain. Starts at 1 and increments
                                               by one  each line. Reset to 1 for each chain.
    12             Character      chainID      Chain identifier. This may be any single
                                               legal  character, including a blank which is
                                               is  used if there is only one chain.
    14 - 17        Integer        numRes       Number of residues in the chain.
                                               This  value is repeated on every record.
    20 - 22        Residue name   resName      Residue name.
    24 - 26        Residue name   resName      Residue name.
    28 - 30        Residue name   resName      Residue name.
    32 - 34        Residue name   resName      Residue name.
    36 - 38        Residue name   resName      Residue name.
    40 - 42        Residue name   resName      Residue name.
    44 - 46        Residue name   resName      Residue name.
    48 - 50        Residue name   resName      Residue name.
    52 - 54        Residue name   resName      Residue name.
    56 - 58        Residue name   resName      Residue name.
    60 - 62        Residue name   resName      Residue name.
    64 - 66        Residue name   resName      Residue name.
    68 - 70        Residue name   resName      Residue name.
    """

    num_residues = len(sequence)
    line_index = 1
    residues_in_line = []
    for amino_acid in sequence:
        amino_acid_3 = d1to3[amino_acid] if amino_acid in d1to3 else "UNK"
        residues_in_line.append(amino_acid_3)

        if len(residues_in_line) == 13:
            # DO NOT MODIFY THIS LINE
            line = f"SEQRES {line_index:3d} {chain} {num_residues:4d}  {' '.join(residues_in_line)}"
            line_index += 1
            residues_in_line = []

            header.append(line)

    # add the last line
    if len(residues_in_line) > 0:
        line = f"SEQRES {line_index:3d} {chain} {num_residues:4d}  {' '.join(residues_in_line)}"
        header.append(line)

    # combine the header and the body
    all_lines = header + body

    # write the file
    for i, line in enumerate(all_lines):
        if not line.endswith("\n"):
            all_lines[i] = line + "\n"

    if out_pdb_file is None:
        out_pdb_file = in_pdb_file

    with open(out_pdb_file, "w") as f:
        f.writelines(all_lines)


def mock_missing_info_by_alignment(
        in_pdb_file: str,
        out_pdb_file: str,
        chain: Optional[str] = None,
):
    header = read_pdb_header(in_pdb_file)
    header_sequence = get_sequence_from_header(header)
    structure: AtomArray = read_pdb_file(in_pdb_file)

    if chain is None:
        chain = get_chains(structure)[0]

    chain_structure = structure[structure.chain_id == chain]
    sequence = header_sequence[chain]
    aligned_missing_residues = align_structure_to_sequence(sequence, in_pdb_file)

    missing_residues_ids_aligned = set([res_id for res_id, _ in aligned_missing_residues])
    correct_aa = {
        res_id: aa
        for res_id, aa in aligned_missing_residues
    }

    broken_ids = set(broken_residues_by_structure(structure)[chain]).union(
        set(missing_residues_by_structure_continuity(structure, broken_only=True)[chain])
    )

    all_ids = set(chain_structure.res_id).union(set(missing_residues_ids_aligned)).union(broken_ids)
    all_ids = sorted(list(all_ids))

    new_residues = []
    for res_id in all_ids:
        if res_id not in missing_residues_ids_aligned and res_id not in broken_ids:
            residue = chain_structure[chain_structure.res_id == res_id]
            new_residues.append(residue)
            continue

        # Missing Residue
        if res_id in correct_aa:
            aa = correct_aa[res_id]
            aa3 = d1to3[aa] if aa in d1to3 else "UNK"
        # Broken Residue
        else:
            first_atom = chain_structure[chain_structure.res_id == res_id][0]
            aa3 = first_atom.res_name

        n_atom = Atom(
            coord=np.random.rand(3),
            chain_id=chain,
            res_id=res_id,
            res_name=aa3,
            atom_name="N",
            element="N",
        )

        ca_atom = Atom(
            coord=np.random.rand(3),
            chain_id=chain,
            res_id=res_id,
            res_name=aa3,
            atom_name="CA",
            element="C",
        )

        c_atom = Atom(
            coord=np.random.rand(3),
            chain_id=chain,
            res_id=res_id,
            res_name=aa3,
            atom_name="C",
            element="C",
        )

        residue = [n_atom, ca_atom, c_atom]
        new_residue = struct_array(residue)
        new_residues.append(new_residue)

    atom_arrays = []
    for residue in new_residues:
        atom_arrays.extend(residue)

    new_atom_array = struct_array(atom_arrays)
    write_structure_to_pdb(new_atom_array, out_pdb_file)

    # add old header to new file
    add_header_info(header, out_pdb_file)

    start_indexes = {
        chain: int(min(all_ids))
    }

    # Compute the missing_residues_id
    missing_residues_id = {chain: []}
    missing_residues_id[chain].extend(missing_residues_ids_aligned)
    missing_residues_id[chain].extend(broken_ids)
    missing_residues_id[chain] = sorted(list(set(missing_residues_id[chain])))
    missing_residues_id[chain] = [int(res_id) for res_id in missing_residues_id[chain]]

    # Compute mapping from real res_ids to standard [0, 1, 2, ...]
    mapping = {chain: {}}
    for i, res_id in enumerate(all_ids):
        res_id = int(res_id)
        mapping[chain][res_id] = i

    missing_info = {
        "missing_residues_id": missing_residues_id,
        "start_indexes": start_indexes,
        "index_mapping": mapping
    }

    # write in out_pdb_file + ".missing" the missing residues as a json
    with open(out_pdb_file + ".missing", "w") as f:
        json.dump(missing_info, f)

    return new_atom_array


def main():
    pdb_file = "pdb_to_correct/s1_header/2XRA_A.pdb"
    pdb_file_out = "pdb_to_correct_debug/2XRA_A_mocked.pdb"
    mock_missing_info_by_alignment(pdb_file, pdb_file_out)


if __name__ == "__main__":
    main()