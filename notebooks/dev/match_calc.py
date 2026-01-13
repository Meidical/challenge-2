# Imports
import os
import tarfile
import urllib
import pandas as pd
from pandas import DataFrame
import numpy as np

# Match Utility Functions
# split letter and asterisk to allele 
# compare antigen matching between donor and recipient

def get_HLA_match(donor_tissue_type, recipient_tissue_type):
    """
    Compare HLA alleles between donor and recipient.
    For each locus, count how many donor alleles match any recipient alleles.

    Returns: matches, missing_allell, missing_antigen
    """
    matches, missing_allell, missing_antigen = 0, 0, 0

    for donor_row, recipient_row in zip(donor_tissue_type, recipient_tissue_type):
        donor_row_sorted = sorted(donor_row)
        recipient_row_sorted = sorted(recipient_row)

        for donor_val, recipient_val in zip(donor_row_sorted, recipient_row_sorted):
            if donor_val == recipient_val:
                matches += 1
            else:
                if donor_val.split(':')[0] != recipient_val.split(':')[0]:
                    missing_allell += 1
                if donor_val.split(':')[1] != recipient_val.split(':')[1]:
                    missing_antigen += 1

    print(f"HLA Match: {matches}, Missing Allell: {missing_allell}, Missing Antigen: {missing_antigen}")
    return matches, missing_allell, missing_antigen


# a (absent): 0, P (present): 1
def get_CMV_serostatus(donor_CMV, recipient_CMV):
    SEROSTATUS_MATRIX = np.array([[0, 1],
                                  [2, 3]])

    return SEROSTATUS_MATRIX[recipient_CMV, donor_CMV]


# F: 0, M: 1
def get_gender_match(donor_gender, recipient_gender):
    # F->M
    if donor_gender == 0 and recipient_gender == 1:
        return 0 # gender mismatch
    
    # F->F, M->F, M->M
    return 1 # gender match



# O: 0, A: 1, B: 2, AB: 3
def get_ABO_match(donor_ABO, recipient_ABO):
    BLOOD_COMPATIBILITY_MATRIX = np.array([[1, 0, 0, 0],
                                           [1, 1, 0, 0],
                                           [1, 0, 1, 0],
                                           [1, 1, 1, 1]])

    return BLOOD_COMPATIBILITY_MATRIX[recipient_ABO, donor_ABO]



# Testing
DONOR_TEST = [['A*01:03','A*02:01'],
                ['B*08:01','B*35:01'],
                ['C*04:03','C*07:02'],
                ['DRB1*01:01','DRB1*04:01'],
                ['DQB1*06:02','DQB1*03:02']]
    
RECIPIENT_TEST = [['A*01:01','A*02:01'],
                    ['B*09:02','B*35:01'],
                    ['C*04:03','C*07:02'],
                    ['DRB1*01:01','DRB1*04:01'],
                    ['DQB1*06:02','DQB1*03:02']]

HLA_match, antigen, allel = get_HLA_match(DONOR_TEST, RECIPIENT_TEST)
#print(f'HLA Match: {HLA_match}, Missing Antigen: {antigen}, Missing Allell: {allel}')
