import numpy as np

class MatchUtils:
    @staticmethod
    def get_HLA_match(donor_tissue_type, recipient_tissue_type):
        matches, missing_allell, missing_antigen = 0, 0, 0

        for donor_row, recipient_row in zip(donor_tissue_type, recipient_tissue_type):
            donor_row_sorted = sorted(donor_row)
            recipient_row_sorted = sorted(recipient_row)

            for donor_val, recipient_val in zip(donor_row_sorted, recipient_row_sorted):
                if donor_val == recipient_val:
                    matches += 1
                else:
                    if donor_val.split(":")[0] != recipient_val.split(":")[0]:
                        missing_allell += 1
                    if donor_val.split(":")[1] != recipient_val.split(":")[1]:
                        missing_antigen += 1

        return matches, missing_allell, missing_antigen

    @staticmethod
    def get_CMV_status(donor_CMV, recipient_CMV):
        SEROSTATUS_MATRIX = np.array([[0, 1],
                                    [2, 3]])

        return SEROSTATUS_MATRIX[recipient_CMV, donor_CMV]

    @staticmethod
    def get_gender_match(donor_gender, recipient_gender):
        if donor_gender == 0 and recipient_gender == 1:
            return 0

        return 1

    @staticmethod
    def get_ABO_match(donor_ABO, recipient_ABO):
        BLOOD_COMPATIBILITY_MATRIX = np.array([[1, 0, 0, 0],
                                            [1, 1, 0, 0],
                                            [1, 0, 1, 0],
                                            [1, 1, 1, 1]])

        return BLOOD_COMPATIBILITY_MATRIX[recipient_ABO, donor_ABO]
