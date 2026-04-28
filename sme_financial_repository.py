from types import SimpleNamespace
import pandas as pd
import logging
from config.table_config import TABLE_INFO
from repository.sme_repository import SMERepository

class SMEFinancialRepository(SMERepository):
    def __init__(self):
        super(SMEFinancialRepository, self).__init__()
        col_info = TABLE_INFO[self.name]['columns']

        self.cols = SimpleNamespace(
            ClientId = self.column_struct(col_info.get('ClientId'), 'int64'),
            RefDate = self.column_struct(col_info.get('RefDate'), 'datetime64[ns]'),

            A_1 = self.column_struct(col_info.get('A_1'), "float32", is_amount=True),
            A_10 = self.column_struct(col_info.get('A_10'), "float32", is_amount=True),
            A_11 = self.column_struct(col_info.get('A_11'), "float32", is_amount=True),
            A_13 = self.column_struct(col_info.get('A_13'), "float32", is_amount=True),
            A_131 = self.column_struct(col_info.get('A_131'), "float32", is_amount=True),
            A_132 = self.column_struct(col_info.get('A_132'), "float32", is_amount=True),
            A_15 = self.column_struct(col_info.get('A_15'), "float32", is_amount=True),
            A_150 = self.column_struct(col_info.get('A_150'), "float32", is_amount=True),
            A_151 = self.column_struct(col_info.get('A_151'), "float32", is_amount=True),
            A_17 = self.column_struct(col_info.get('A_17'), "float32", is_amount=True),
            A_2 = self.column_struct(col_info.get('A_2'), "float32", is_amount=True),
            A_22 = self.column_struct(col_info.get('A_22'), "float32", is_amount=True),
            A_231 = self.column_struct(col_info.get('A_231'), "float32", is_amount=True),
            A_232 = self.column_struct(col_info.get('A_232'), "float32", is_amount=True),
            A_24 = self.column_struct(col_info.get('A_24'), "float32", is_amount=True),
            A_253 = self.column_struct(col_info.get('A_253'), "float32", is_amount=True),
            A_258 = self.column_struct(col_info.get('A_258'), "float32", is_amount=True),
            A_29 = self.column_struct(col_info.get('A_29'), "float32", is_amount=True),

            P_3 = self.column_struct(col_info.get('P_3'), "float32", is_amount=True),
            P_30 = self.column_struct(col_info.get('P_30'), "float32", is_amount=True),
            P_300 = self.column_struct(col_info.get('P_300'), "float32", is_amount=True),
            P_331 = self.column_struct(col_info.get('P_331'), "float32", is_amount=True),
            P_340 = self.column_struct(col_info.get('P_340'), "float32", is_amount=True),
            P_35 = self.column_struct(col_info.get('P_35'), "float32", is_amount=True),
            P_4 = self.column_struct(col_info.get('P_4'), "float32", is_amount=True),
            P_40 = self.column_struct(col_info.get('P_40'), "float32", is_amount=True),
            P_400 = self.column_struct(col_info.get('P_400'), "float32", is_amount=True),
            P_42 = self.column_struct(col_info.get('P_42'), "float32", is_amount=True),
            P_431 = self.column_struct(col_info.get('P_431'), "float32", is_amount=True),
            P_440 = self.column_struct(col_info.get('P_440'), "float32", is_amount=True),
            P_472 = self.column_struct(col_info.get('P_472'), "float32", is_amount=True),
            P_5 = self.column_struct(col_info.get('P_5'), "float32", is_amount=True),

            G_60 = self.column_struct(col_info.get('G_60'), "float32", is_amount=True),
            G_600 = self.column_struct(col_info.get('G_600'), "float32", is_amount=True),
            G_601 = self.column_struct(col_info.get('G_601'), "float32", is_amount=True),
            G_602 = self.column_struct(col_info.get('G_602'), "float32", is_amount=True),
            G_610 = self.column_struct(col_info.get('G_610'), "float32", is_amount=True),
            G_611 = self.column_struct(col_info.get('G_611'), "float32", is_amount=True),
            G_612 = self.column_struct(col_info.get('G_612'), "float32", is_amount=True),
            G_62 = self.column_struct(col_info.get('G_62'), "float32", is_amount=True),
            G_63 = self.column_struct(col_info.get('G_63'), "float32", is_amount=True),
            G_642 = self.column_struct(col_info.get('G_642'), "float32", is_amount=True),
            G_646 = self.column_struct(col_info.get('G_646'), "float32", is_amount=True),
            G_656 = self.column_struct(col_info.get('G_656'), "float32", is_amount=True),
            G_66 = self.column_struct(col_info.get('G_66'), "float32", is_amount=True),
            G_690 = self.column_struct(col_info.get('G_690'), "float32", is_amount=True),
            G_691 = self.column_struct(col_info.get('G_691'), "float32", is_amount=True),
            G_692 = self.column_struct(col_info.get('G_692'), "float32", is_amount=True),

            Amortisman_ve_Itfa_Giderleri = self.column_struct(col_info.get('Amortisman_ve_Itfa_Giderleri'), "float32", is_amount=True),

            A_12 = self.column_struct(col_info.get('A_12'), "float32", is_amount=True),
            A_23 = self.column_struct(col_info.get('A_23'), "float32", is_amount=True),
            A_25 = self.column_struct(col_info.get('A_25'), "float32", is_amount=True),
            A_26 = self.column_struct(col_info.get('A_26'), "float32", is_amount=True),
            P_32 = self.column_struct(col_info.get('P_32'), "float32", is_amount=True),
            P_34 = self.column_struct(col_info.get('P_34'), "float32", is_amount=True),
            P_43 = self.column_struct(col_info.get('P_43'), "float32", is_amount=True),
            P_47 = self.column_struct(col_info.get('P_47'), "float32", is_amount=True)
        )

        self.mapped_keys = ['ClientId', 'RefDate']

    
    def get_all(self, filter_dates) -> pd.DataFrame:
        """
        Fetches all columns from the data source.

        Returns:
            pd.DataFrame: Table fetched from the data source.
        """

        columns_list = []
        for column, column_info in self.cols.__dict__.items():
            column_str = ""

            if "+" in column_info.n:
                added_columns = column_info.n.split(" + ")

                for added_column in added_columns:
                    column_str += f"[{added_column}] + "

                column_str = column_str[:-3] + f" as {column}"
            else:
                column_str = f"[{column_info.n}]"

            columns_list.append(column_str)

        column_str = ", ".join(columns_list)

        sql = f"SELECT {column_str} FROM {self.db_config.database}.{self.schema}.{self.table_name} WITH(NOLOCK)"

        if len(filter_dates) > 0:
            min_filter_date, max_filter_date = str(filter_dates[0]), str(filter_dates[1])
            min_filter_date = pd.to_datetime(min_filter_date, format='%Y%m%d')
            min_filter_date_minus_29_months = min_filter_date - pd.DateOffset(months=29)
            min_filter_date_minus_29_months = min_filter_date_minus_29_months.strftime('%Y%m%d')

            ref_date_db_name = self.cols.__dict__['RefDate'].n
            filter_clause = f""" WHERE
            ({ref_date_db_name} >= CAST('{min_filter_date_minus_29_months}' AS DATE))
            AND ({ref_date_db_name} <= CAST('{max_filter_date}' AS DATE))
            """

            sql += filter_clause

        sql += " AND MONTH(FIN_DATE) = 12"

        logging.info(f'Reading {self.name}')
        logging.info(f'Running query: {sql}')

        df = self.run_query(sql)
        print(df.shape)

        if "RefDate" in df.columns:
            df["RefDate"] = df["RefDate"].dt.normalize()

        df = df.sort_values(self.mapped_keys)
        return df
