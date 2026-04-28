import logging
import pandas as pd
from feature_processors.base_feature_processor import BaseFeatureProcessor
from repository.sme_financial_repository import SMEFinancialRepository
from pathlib import Path
import yaml
import json
import numpy as np

from utils.helper import normalize_amount_columns

class SMEFinancialFeatureProcessor(BaseFeatureProcessor):
    def __init__(self, config):
        super(SMEFinancialFeatureProcessor, self).__init__(config, SMEFinancialRepository())
        self.config = config
        self.repository = SMEFinancialRepository()
        self.rds_key_cols = ["ClientId", "FIN_DATE"]

        # Update function map for ratio calculation
        self.update_function_map({"calculate_financial_aggregate": self.calculate_financial_aggregate})
        self.update_function_map({"calculate_financial_ratio": self.calculate_financial_ratio})

        self.update_function_map({"calculate_custom_financial_ratio": self.calculate_custom_financial_ratio})
        self.custom_financial_ratio_function_map = {
            "DC_31": self.__calculate_DC_31,
            "EF_13": self.__calculate_EF_13,
            "EF_14": self.__calculate_EF_14,
            "EF_17": self.__calculate_EF_17,
            "EF_19": self.__calculate_EF_19,
            "EF_20": self.__calculate_EF_20,
            "FS_12": self.__calculate_FS_12,
            "FS_53_CONT": self.__calculate_FS_53_CONT,
            "FS_55_CONT": self.__calculate_FS_55_CONT,
            "FS_56_CONT": self.__calculate_FS_56_CONT,
            "LQ_13": self.__calculate_LQ_13,
            "LQ_14": self.__calculate_LQ_14,
            "LQ_50": self.__calculate_LQ_50,
            "PROF_32": self.__calculate_PROF_32,
            "PROF_33": self.__calculate_PROF_33,
            "TR_01": self.__calculate_TR_01,
            "TR_04": self.__calculate_TR_04,
            "TR_05": self.__calculate_TR_05,
            "TR_06": self.__calculate_TR_06,
            "TR_07": self.__calculate_TR_07,
            "TR_08": self.__calculate_TR_08,
            "TR_09": self.__calculate_TR_09,
            "TR_10": self.__calculate_TR_10,
            "TR_11": self.__calculate_TR_11,
            "TR_12": self.__calculate_TR_12,
            "TR_13": self.__calculate_TR_13,
            "TR_14": self.__calculate_TR_14,
            "TR_15": self.__calculate_TR_15,
            "TR_17": self.__calculate_TR_17,
            "TR_18": self.__calculate_TR_18,
            "TR_19": self.__calculate_TR_19,
            "CONST_06": self.__calculate_CONST_06,
            "CONST_07": self.__calculate_CONST_07,
            "CONST_09": self.__calculate_CONST_09,
            "CONST_11": self.__calculate_CONST_11,
            "CONST_17": self.__calculate_CONST_17,
            "CONST_18": self.__calculate_CONST_18,
            "CONST_20": self.__calculate_CONST_20,
            "CONST_22": self.__calculate_CONST_22,
            "CONST_FS_12": self.__calculate_CONST_FS_12,
            "CONST_TR_02": self.__calculate_CONST_TR_02
        }

    def resample(self, df: pd.DataFrame) -> pd.DataFrame:
        # get date and max date in the data
        date_filter = [df.RefDate.min(), df.RefDate.max()]

        possible_months = pd.Series(
            pd.date_range(start=date_filter[0], end=date_filter[1], freq="Y").strftime("%Y-%m-%d"),
            name="RefDate"
        ).to_frame()
        possible_months["RefDate"] = pd.to_datetime(possible_months["RefDate"])

        # get unique customer id's
        possible_customers = pd.DataFrame({"ClientId": df["ClientId"].unique()})

        # get all combinations for possible couples of RefDate and ClientIds
        resampled_df = possible_customers.merge(possible_months, how="cross")
        resampled_df = resampled_df.merge(df, on=["ClientId", "RefDate"], how="left")

        return resampled_df

    def fetch_columns(self, columns: list) -> pd.DataFrame:

        if self.repository is None:
            logging.warning(
                f"{self.name} does not support raw feature fetch. Possibly due top multiple repo dependencies.")
            return None

        # fetch the all data from database and save after resampling and normalization
        if self.save_preprocessed_data:

            df = self.repository.get_all(filter_dates=self.config.feature_generation.filter_dates)

            df = self.resample(df)

            if self.config.feature_generation.apply_normalization:
                col_map = {k: v.is_amount for k, v in self.repository.cols.__dict__.items()}
                normalization_type = self.config.feature_generation.normalization_type
                base_normalization_month = self.config.feature_generation.base_normalization_month
                df = normalize_amount_columns(
                    df,
                    col_map,
                    normalization_type,
                    base_normalization_month
                )

            Path.joinpath(self.repository.db.config.file_path).mkdir(parents=True, exist_ok=True)
            all_data_path = Path.joinpath(self.repository.db.config.file_path, self.repository.table_name + '.parquet')

            df.to_parquet(all_data_path)
            self.save_preprocessed_data = False
            del df

        df = self.get_preprocessed_data(columns)
        df = df.sort_values(by=['ClientId', 'RefDate'])

        return df

    def calculate_financial_aggregate(
            self,
            column_name: str,
            added_accounts: list = [],
            subtracted_accounts: list = []
    ):
        # Return directly if the financial aggregate has already been calculated
        if self.generated_data is not None and \
                column_name in self.generated_data.columns:
            return self.generated_data[self.key_cols + [column_name]]

        df = self.fetch_columns(added_accounts + subtracted_accounts)

        added_accounts_sum = df[added_accounts].sum(axis=1) if added_accounts else 0
        subtracted_accounts_sum = df[subtracted_accounts].sum(axis=1) if subtracted_accounts else 0

        df_financial_aggregates = df[["ClientId", "RefDate"]].copy()
        df_financial_aggregates[column_name] = added_accounts_sum - subtracted_accounts_sum

        df_financial_aggregates[column_name] = df_financial_aggregates[column_name].astype("Float32")

        if self.generated_data is None:
            self.generated_data = df_financial_aggregates
        elif column_name not in self.generated_data.columns:
            self.generated_data = self.generated_data.merge(df_financial_aggregates, on=self.key_cols, how="left")

        return df_financial_aggregates

    def calculate_financial_ratio(
            self,
            column_name: str,
            numerator_column_name: str,
            denominator_column_name: str
    ):
        # Return directly if the ratio has already been calculated
        if self.generated_data is not None and \
                column_name in self.generated_data.columns:
            return self.generated_data[self.key_cols + [column_name]]

        def _get_or_compute_financial_aggregate(financial_aggregate: str) -> pd.DataFrame:
            # Return directly if the financial aggregate has already been calculated
            if self.generated_data is not None and financial_aggregate in self.generated_data.columns:
                return self.generated_data[self.key_cols + [financial_aggregate]]

            if financial_aggregate not in self.longlist:
                raise KeyError(f'"{financial_aggregate}" not found in longlist. '
                               f"Cannot compute for ratio '{column_name}'.")

            func = self.longlist[financial_aggregate]["func"]
            args = self.longlist[financial_aggregate]["args"]
            return self.calculate_multiple_features_from_YAML(func, **args)

        numerator = _get_or_compute_financial_aggregate(numerator_column_name)
        denominator = _get_or_compute_financial_aggregate(denominator_column_name)

        merged_data = numerator.merge(denominator, on=self.key_cols, how="outer")

        ratio = merged_data[numerator_column_name].div(merged_data[denominator_column_name])
        ratio = ratio.replace([np.inf, -np.inf], np.nan)

        merged_data[column_name] = ratio.astype("Float32")

        feature = merged_data[self.key_cols + [column_name]].copy()

        if self.generated_data is None:
            self.generated_data = feature
        elif column_name not in self.generated_data.columns:
            self.generated_data = self.generated_data.merge(feature, on=self.key_cols, how="left")

        return feature

    def __fetch_required_columns_and_shift(self, required_columns, columns_to_shift):
        df = self.fetch_columns(required_columns)

        shifted_column_names = []
        for column_to_shift in columns_to_shift:
            shifted_column_names.append(f"{column_to_shift}_prev")

        df[shifted_column_names] = df.groupby("ClientId")[columns_to_shift].shift(1)

        return df
    
    def calculate_custom_financial_ratio(
            self,
            column_name: str,
            required_columns: list,
            columns_to_shift: list = []
    ):
        # Return directly if the ratio has already been calculated
        if self.generated_data is not None and \
                column_name in self.generated_data.columns:
            return self.generated_data[self.key_cols + [column_name]]

        df = self.__fetch_required_columns_and_shift(required_columns, columns_to_shift)

        df[column_name] = self.custom_financial_ratio_function_map[column_name](df)

        df[column_name] = df[column_name].replace([np.inf, -np.inf], np.nan)
        df[column_name] = df[column_name].astype("Float32")

        feature = df[self.key_cols + [column_name]].copy()

        if self.generated_data is None:
            self.generated_data = feature
        elif column_name not in self.generated_data.columns:
            self.generated_data = self.generated_data.merge(feature, on=self.key_cols, how="left")

        return feature

    def __calculate_DC_31(self, df):
        numerator = (
                df["G_692"] + df["G_66"] + df["G_691"] + df["Amortisman_ve_Itfa_Giderleri"]
                + df["G_692_prev"] + df["G_66_prev"] + df["G_691_prev"] + df["Amortisman_ve_Itfa_Giderleri_prev"]
        ) / 2

        denominator = df["P_3"] - df["P_31"]

        return numerator / denominator

    def __calculate_EF_13(self, df):
        numerator = df["G_0"] + df["A_15"] - df["A_15_prev"]

        denominator = df["A_15"]

        return numerator / denominator

    def __calculate_EF_14(self, df):
        numerator = (
                df["G_600"] + df["G_601"] + df["G_602"]
                - (df["G_610"] + df["G_611"] + df["G_612"])
                + df["A_15"] - df["A_15_prev"]
        )

        denominator = df["A_15"]

        return numerator / denominator

    def __calculate_EF_17(self, df):
        numerator = 365 * (df["A_15"] + df["A_15_prev"]) / 2

        denominator = df["G_62"]

        return numerator / denominator

    def __calculate_EF_19(self, df):
        numerator = 365 * ((df["P_32"] + df["P_42"] + df["P_32_prev"] + df["P_42_prev"]) / 2)

        denominator = df["G_62"]

        return numerator / denominator

    def __calculate_EF_20(self, df):
        numerator = 365 * ((df["A_12"] + df["A_22"] + df["A_12_prev"] + df["A_22_prev"]) / 2)

        denominator = (
                df["G_600"] + df["G_601"] + df["G_602"]
                - (df["G_610"] + df["G_611"] + df["G_612"])
        )

        return numerator / denominator

    def __calculate_FS_12(self, df):
        numerator = (df["P_30"] - df["P_30_prev"] + df["P_40"] - df["P_40_prev"])

        denominator = df["A_1"] + df["A_2"]

        return numerator / denominator

    def __calculate_FS_53_CONT(self, df):
        numerator = (
                df["G_600"] + df["G_601"] + df["G_602"]
                - (df["G_610"] + df["G_611"] + df["G_612"])
                + df["P_35"] + df["P_35_prev"]
        ) / 2 + df["P_34"]

        denominator = (df["A_17"] + df["A_17_prev"]) / 2 + df["A_15"] + df["G_62"]

        return numerator / denominator

    def __calculate_FS_55_CONT(self, df):
        numerator = (df["P_35"] + df["P_35_prev"]) / 2

        denominator = (df["A_17"] + df["A_17_prev"]) / 2

        return numerator / denominator

    def __calculate_FS_56_CONT(self, df):
        numerator = (
            df["G_600"] + df["G_601"] + df["G_602"]
            - (df["G_610"] + df["G_611"] + df["G_612"])
            + df["G_600_prev"] + df["G_601_prev"] + df["G_602_prev"]
            - (df["G_610_prev"] + df["G_611_prev"] + df["G_612_prev"])
            + df["P_35"] + df["P_35_prev"]
        ) / 2

        denominator = (df["A_17"] + df["A_17_prev"] + df["G_62"] + df["G_62_prev"]) / 2

        return numerator / denominator


    def __calculate_LQ_13(self, df):
        numerator = df["A_1"] - df["P_3"]

        denominator = df["G_60"] + df["A_15"] - df["A_15_prev"]

        return numerator / denominator


    def __calculate_LQ_14(self, df):
        numerator = df["A_1"] - df["P_3"]

        denominator = (
            df["G_600"] + df["G_601"] + df["G_602"]
            - (df["G_610"] + df["G_611"] + df["G_612"])
            + df["A_15"] - df["A_15_prev"]
        )

        return numerator / denominator


    def __calculate_LQ_50(self, df):
        numerator = df["A_12"] + df["A_10"] + df["A_11"] + (df["A_15"] + df["A_15_prev"]) / 2

        denominator = df["P_3"] - df["P_331"]

        return numerator / denominator


    def __calculate_PROF_32(self, df):
        numerator = (
            df["G_692"] + df["G_66"] + df["G_691"] + df["Amortisman_ve_Itfa_Giderleri"]
            - (df["A_25"] - df["A_25_prev"] + df["Amortisman_ve_Itfa_Giderleri"])
        )

        denominator = df["G_66"] - df["G_642"]

        return numerator / denominator


    def __calculate_PROF_33(self, df):
        numerator = (
            df["G_692"] + df["G_66"] + df["G_691"] + df["Amortisman_ve_Itfa_Giderleri"]
            - (df["A_25"] - df["A_25_prev"] + df["Amortisman_ve_Itfa_Giderleri"])
        )

        denominator = df["P_30"] + df["P_40"] + df["G_66"] - df["A_10"]

        return numerator / denominator


    def __calculate_TR_01(self, df):
        numerator = df["G_60"] - df["G_60_prev"]

        denominator = df["G_60_prev"]

        return numerator / denominator


    def __calculate_TR_04(self, df):
        numerator = df["G_692"] - df["G_692_prev"]

        denominator = df["G_692_prev"]

        return numerator / denominator


    def __calculate_TR_05(self, df):
        numerator = (
            df["G_692"] + df["G_66"] + df["G_691"] + df["Amortisman_ve_Itfa_Giderleri"]
            - (df["G_692_prev"] + df["G_66_prev"] + df["G_691_prev"] + df["Amortisman_ve_Itfa_Giderleri_prev"])
        )

        denominator = df["G_692_prev"] + df["G_66_prev"] + df["G_691_prev"] + df["Amortisman_ve_Itfa_Giderleri_prev"]

        return numerator / denominator


    def __calculate_TR_06(self, df):
        numerator = df["P_5"] - df["P_5_prev"]

        denominator = df["P_5_prev"]

        return numerator / denominator


    def __calculate_TR_07(self, df):
        numerator = df["P_3"] - df["P_3_prev"]

        denominator = df["P_3_prev"]

        return numerator / denominator
    
    def __calculate_TR_08(self, df):
        numerator = (
            df["G_600"] + df["G_601"] + df["G_602"]
            - (df["G_610"] + df["G_611"] + df["G_612"])
            + df["G_62"]
            - (
                df["G_600_prev"] + df["G_601_prev"] + df["G_602_prev"]
                - (df["G_610_prev"] + df["G_611_prev"] + df["G_612_prev"])
                + df["G_62_prev"]
            )
        )

        denominator = (
            df["G_600_prev"] + df["G_601_prev"] + df["G_602_prev"]
            - (df["G_610_prev"] + df["G_611_prev"] + df["G_612_prev"])
            + df["G_62_prev"]
        )

        return numerator / denominator


    def __calculate_TR_09(self, df):
        numerator = (
            df["G_600"] + df["G_601"] + df["G_602"]
            - (df["G_610"] + df["G_611"] + df["G_612"])
            - (
                df["G_600_prev"] + df["G_601_prev"] + df["G_602_prev"]
                - (df["G_610_prev"] + df["G_611_prev"] + df["G_612_prev"])
            )
        )

        denominator = (
            df["G_600_prev"] + df["G_601_prev"] + df["G_602_prev"]
            - (df["G_610_prev"] + df["G_611_prev"] + df["G_612_prev"])
        )

        return numerator / denominator


    def __calculate_TR_10(self, df):
        numerator = df["A_1"] + df["A_2"] - (df["A_1_prev"] + df["A_2_prev"])

        denominator = df["A_1_prev"] + df["A_2_prev"]

        return numerator / denominator


    def __calculate_TR_11(self, df):
        numerator = df["A_1"] + df["P_3"] - (df["A_1_prev"] + df["P_3_prev"])

        denominator = df["A_1_prev"] + df["P_3_prev"]

        return numerator / denominator


    def __calculate_TR_12(self, df):
        numerator = df["P_4"] - df["P_4_prev"]

        denominator = df["P_4_prev"]

        return numerator / denominator


    def __calculate_TR_13(self, df):
        numerator = df["A_25"] - df["A_25_prev"]

        denominator = df["A_25_prev"]

        return numerator / denominator


    def __calculate_TR_14(self, df):
        numerator = df["A_2"] - df["A_2_prev"]

        denominator = df["A_2_prev"]

        return numerator / denominator


    def __calculate_TR_15(self, df):
        numerator = (
            df["A_12"]
            - (
                df["G_600"] + df["G_601"] + df["G_602"]
                - (df["G_610"] + df["G_611"] + df["G_612"])
            )
            - (
                df["A_12_prev"]
                - (
                    df["G_600_prev"] + df["G_601_prev"] + df["G_602_prev"]
                    - (df["G_610_prev"] + df["G_611_prev"] + df["G_612_prev"])
                )
            )
        )

        denominator = (
            df["A_12_prev"]
            - (
                df["G_600_prev"] + df["G_601_prev"] + df["G_602_prev"]
                - (df["G_610_prev"] + df["G_611_prev"] + df["G_612_prev"])
            )
        )

        return numerator / denominator


    def __calculate_TR_17(self, df):
        numerator = df["P_30"] + df["P_40"] - (df["P_30_prev"] + df["P_40_prev"])

        denominator = df["P_30_prev"] + df["P_40_prev"]

        return numerator / denominator


    def __calculate_TR_18(self, df):
        numerator = df["P_30"] - df["P_30_prev"]

        denominator = df["P_30_prev"]

        return numerator / denominator
    
    def __calculate_TR_19(self, df):
        numerator = df["P_40"] - df["P_40_prev"]

        denominator = df["P_40_prev"]

        return numerator / denominator


    def __calculate_CONST_06(self, df):
        numerator = (
            df["P_35"] - df["A_17"]
            - (df["P_35_prev"] - df["A_17_prev"])
            + df["G_692"] + df["G_66"] + df["G_691"] + df["Amortisman_ve_Itfa_Giderleri"]
        )

        denominator = df["G_66"]

        return numerator / denominator


    def __calculate_CONST_07(self, df):
        numerator = (
            df["P_35"] - df["A_17"]
            - (df["P_35_prev"] - df["A_17_prev"])
            + df["G_692"] + df["G_66"] + df["G_691"] + df["Amortisman_ve_Itfa_Giderleri"]
            - (df["A_12"] - df["A_12_prev"])
            + (df["P_32"] - df["P_32_prev"])
        )

        denominator = df["P_30"] + df["G_66"]

        return numerator / denominator


    def __calculate_CONST_09(self, df):
        numerator = (
            df["P_35"] - df["A_17"]
            - (df["P_35_prev"] - df["A_17_prev"])
            + df["G_692"] + df["G_66"] + df["G_691"] + df["Amortisman_ve_Itfa_Giderleri"]
        )

        denominator = df["P_3"] + df["P_40"] + df["P_42"] + df["P_43"]

        return numerator / denominator


    def __calculate_CONST_11(self, df):
        numerator = (
            df["P_35"] - df["A_17"]
            - (df["P_35_prev"] - df["A_17_prev"])
            + df["G_692"] + df["G_66"] + df["G_691"]
        )

        denominator = df["P_3"] + df["P_40"] + df["P_42"] + df["P_43"] - df["P_35"]

        return numerator / denominator


    def __calculate_CONST_17(self, df):
        numerator = (
            df["P_35"] - df["A_17"]
            - (df["P_35_prev"] - df["A_17_prev"])
            + df["G_692"] + df["G_66"] + df["G_691"] + df["Amortisman_ve_Itfa_Giderleri"]
            + df["G_600"] + df["G_601"] + df["G_602"]
            - (df["G_610"] + df["G_611"] + df["G_612"])
            - (
                df["G_600_prev"] + df["G_601_prev"] + df["G_602_prev"]
                - (df["G_610_prev"] + df["G_611_prev"] + df["G_612_prev"])
            )
        )

        denominator = df["G_66"]

        return numerator / denominator


    def __calculate_CONST_18(self, df):
        numerator = (
            df["P_35"] - df["A_17"]
            - (df["P_35_prev"] - df["A_17_prev"])
            + df["G_692"] + df["G_66"] + df["G_691"] + df["Amortisman_ve_Itfa_Giderleri"]
            + df["G_600"] + df["G_601"] + df["G_602"]
            - (df["G_610"] + df["G_611"] + df["G_612"])
            - (
                df["G_600_prev"] + df["G_601_prev"] + df["G_602_prev"]
                - (df["G_610_prev"] + df["G_611_prev"] + df["G_612_prev"])
            )
            - (df["A_12"] - df["A_12_prev"])
            + (df["P_32"] - df["P_32_prev"])
        )

        denominator = df["P_30"] + df["G_66"]

        return numerator / denominator

    def __calculate_CONST_20(self, df):
        numerator = (
            df["P_35"] - df["A_17"]
            - (df["P_35_prev"] - df["A_17_prev"])
            + df["G_692"] + df["G_66"] + df["G_691"] + df["Amortisman_ve_Itfa_Giderleri"]
            + df["G_600"] + df["G_601"] + df["G_602"]
            - (df["G_610"] + df["G_611"] + df["G_612"])
            - (
                df["G_600_prev"] + df["G_601_prev"] + df["G_602_prev"]
                - (df["G_610_prev"] + df["G_611_prev"] + df["G_612_prev"])
            )
        )

        denominator = df["P_3"] + df["P_40"] + df["P_42"] + df["P_43"]

        return numerator / denominator


    def __calculate_CONST_22(self, df):
        numerator = (
            df["P_35"] - df["A_17"]
            - (df["P_35_prev"] - df["A_17_prev"])
            + df["G_692"] + df["G_66"] + df["G_691"] + df["Amortisman_ve_Itfa_Giderleri"]
            + df["G_600"] + df["G_601"] + df["G_602"]
            - (df["G_610"] + df["G_611"] + df["G_612"])
            - (
                df["G_600_prev"] + df["G_601_prev"] + df["G_602_prev"]
                - (df["G_610_prev"] + df["G_611_prev"] + df["G_612_prev"])
            )
        )

        denominator = df["P_3"] + df["P_40"] + df["P_42"] + df["P_43"] - df["P_35"]

        return numerator / denominator


    def __calculate_CONST_FS_12(self, df):
        numerator = df["P_30"] - df["P_30_prev"] + df["P_40"] - df["P_40_prev"]

        denominator = df["A_1"] + df["A_2"] - df["A_17"]

        return numerator / denominator


    def __calculate_CONST_TR_02(self, df):
        numerator = (
            df["A_1"] + df["A_2"] - df["A_17"]
            - (df["A_1_prev"] + df["A_2_prev"] - df["A_17_prev"])
        )

        denominator = df["A_1_prev"] + df["A_2_prev"] - df["A_17_prev"]

        return numerator / denominator