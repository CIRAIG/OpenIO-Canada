"""
Class that creates symmetric Input-Output tables based on the Supply and Use economic tables provided by Statistic
Canada, available here: https://www150.statcan.gc.ca/n1/en/catalogue/15-602-X
Multiple transformation models are available (industry technology assumption, fixed industry sales structure, etc.) and
the type of classification (productxproduct, or industryxindustry) can be selected as well.
Also produces environmental extensions for the symmetric tables generated based on data from the NPRI found here:
https://open.canada.ca/data/en/dataset/1fb7d8d4-7713-4ec6-b957-4a882a84fed3
"""

import pandas as pd
import numpy as np
import re


class IOTables:
    def __init__(self, supply_use_excel_path, NPRI_excel, classification, assumption, aggregate_final_demand=True):
        """

        :param supply_use_excel_path: the path to the SUT excel file (e.g. /../Detail level/CA_SUT_C2016_D.xlsx)
        :param NPRI_excel: the path to the NPRI excel file (e.g. /../2017_INRP-NPRI.xlsx)
        :param classification: [string] the type of classification to adopt for the symmetric IOT ("product" or "industry")
        :param assumption: [string] the assumption used to create the symmetric IO tables ("industry technology" or
                            "fixed industry sales structure")
        :param aggregate_final_demand: [boolean] aggregating the final demand to 6 elements or not

        """
        self.SU_tables = pd.read_excel(supply_use_excel_path, None)
        self.NPRI = pd.read_excel(NPRI_excel, None)
        self.classification = classification
        self.assumption = assumption

        self.V = pd.DataFrame()
        self.U = pd.DataFrame()
        self.A = pd.DataFrame()
        self.Z = pd.DataFrame()
        self.S = pd.DataFrame()
        self.Y = pd.DataFrame()
        self.SY = pd.DataFrame()
        self.g = pd.DataFrame()
        self.q = pd.DataFrame()
        self.F = pd.DataFrame()

        self.format_tables()
        self.gimme_symmetric_iot()
        if aggregate_final_demand:
            self.aggregate_final_demand()
        self.remove_codes()
        self.produce_environmental_extensions()

    def format_tables(self):
        """
        Extracts the relevant dataframes from the Excel file
        :return: the relevant dataframes
        """

        Supply_table = self.SU_tables['Supply'].copy()
        Use_table = self.SU_tables['Use_Basic'].copy()

        industries = []
        for i in range(0, len(Supply_table.columns)):
            if Supply_table.iloc[11, i] == 'Total':
                break
            if Supply_table.iloc[11, i] not in [np.nan, 'Industries']:
                # tuple with code + name (need code to deal with duplicate names in detailed levels)
                industries.append((Supply_table.iloc[12, i], Supply_table.iloc[11, i]))

        commodities = []
        factors_of_production = []
        for i, element in enumerate(Supply_table.iloc[:, 0].tolist()):
            if type(element) == str:
                # identify by their codes
                if re.search(r'^[M,F,N,G,I,E]\w*\d', element):
                    commodities.append((element, Supply_table.iloc[i, 1]))
                elif re.search(r'^P\w*\d', element) or re.search(r'^GVA', element):
                    factors_of_production.append((element, Supply_table.iloc[i, 1]))

        final_demand = []
        for i in range(0, len(Use_table.columns)):
            if Use_table.iloc[11, i] == 'Total use':
                break
            if Use_table.iloc[11, i] not in [np.nan, 'Industries']:
                final_demand.append((Use_table.iloc[12, i], Use_table.iloc[11, i]))
        final_demand = [i for i in final_demand if i not in industries and i[1] != 'Total']

        df = Supply_table.iloc[14:, 2:]
        df.index = list(zip(Supply_table.iloc[14:, 0].tolist(), Supply_table.iloc[14:, 1].tolist()))
        df.columns = list(zip(Supply_table.iloc[12, 2:].tolist(), Supply_table.iloc[11, 2:].tolist()))
        Supply_table = df

        df = Use_table.iloc[14:, 2:]
        df.index = list(zip(Use_table.iloc[14:, 0].tolist(), Use_table.iloc[14:, 1].tolist()))
        df.columns = list(zip(Use_table.iloc[12, 2:].tolist(), Use_table.iloc[11, 2:].tolist()))
        Use_table = df

        # fill with zeros
        Supply_table.replace('.', 0, inplace=True)
        Use_table.replace('.', 0, inplace=True)

        # get strings as floats
        Supply_table = Supply_table.astype('float64')
        Use_table = Use_table.astype('float64')

        # check calculated totals matched displayed totals
        assert np.allclose(Use_table.iloc[:, Use_table.columns.get_loc(('TOTAL', 'Total'))],
                           Use_table.iloc[:, :Use_table.columns.get_loc(('TOTAL', 'Total'))].sum(axis=1))
        assert np.allclose(Supply_table.iloc[Supply_table.index.get_loc(('TOTAL', 'Total'))],
                           Supply_table.iloc[:Supply_table.index.get_loc(('TOTAL', 'Total'))].sum())

        self.S = Use_table.loc[factors_of_production, industries]
        self.S.drop(('GVA', 'Gross value-added at basic prices'), inplace=True)
        self.Y = Use_table.loc[commodities, final_demand]
        self.SY = Use_table.loc[factors_of_production, final_demand]
        self.SY.drop(('GVA', 'Gross value-added at basic prices'), inplace=True)
        self.g = Supply_table.loc[[('TOTAL', 'Total')], industries]
        self.q = Supply_table.loc[commodities, [('TOTAL', 'Total')]]
        self.V = Supply_table.loc[commodities, industries]
        self.U = Use_table.loc[commodities, industries]

    def gimme_symmetric_iot(self):
        """
        Transforms Supply and Use Tables to symmetric IO tables
        :return: A, Z and S, symmetric IO tables
        """
        inv_q = pd.DataFrame(np.diag((1 / self.q.iloc[:, 0]).replace(np.inf, 0)), self.q.index, self.q.index)
        inv_g = pd.DataFrame(np.diag((1 / self.g.iloc[0]).replace(np.inf, 0)), self.g.columns, self.g.columns)

        if self.assumption == "industry technology" and self.classification == "product":
            self.Z = self.U.dot(inv_g.dot(self.V.T))
            self.A = self.U.dot(inv_g.dot(self.V.T)).dot(inv_q)
            self.S = self.S.dot(inv_g.dot(self.V.T)).dot(inv_q)
        elif self.assumption == "fixed industry sales structure" and self.classification == "industry":
            self.Z = self.V.T.dot(inv_q).dot(self.U)
            self.A = self.V.T.dot(inv_q).dot(self.U).dot(inv_g)
            self.S = self.S.dot(inv_g)

    def aggregate_final_demand(self):
        """
        Aggregates all final demand sectors into 6 elements: ["Household final consumption expenditure",
        "Non-profit institutions serving households' final consumption expenditure",
        "Governments final consumption expenditure", "Gross fixed capital formation", "Changes in inventories",
        "International exports"]
        :return: the aggregated final demands
        """
        # final demands are identified through their codes, hence the use of regex
        self.Y.loc[:, "Household final consumption expenditure"] = self.Y.loc[:, [i for i in self.Y.columns if
                                                                        re.search(r'^PEC\w*\d', i[0])]].sum(axis=1)
        self.Y.loc[:, "Non-profit institutions serving households' final consumption expenditure"] = self.Y.loc[:,
                                                                                                [i for i in self.Y.columns if
                                                                                                 re.search(r'^CEN\w*\d',
                                                                                                           i[0])]].sum(
            axis=1)
        self.Y.loc[:, "Governments final consumption expenditure"] = self.Y.loc[:, [i for i in self.Y.columns if
                                                                          re.search(r'^CEG\w*\d', i[0])]].sum(axis=1)
        self.Y.loc[:, "Gross fixed capital formation"] = self.Y.loc[:, [i for i in self.Y.columns if
                                                              re.search(r'^CO\w*\d|^ME\w*\d|^IP\w*\d', i[0])]].sum(
            axis=1)
        self.Y.loc[:, "Changes in inventories"] = self.Y.loc[:, [i for i in self.Y.columns if re.search(r'^INV', i[0])]].sum(axis=1)
        self.Y.loc[:, "International exports"] = self.Y.loc[:, [i for i in self.Y.columns if re.search(r'^INT', i[0])]].sum(axis=1)
        self.Y.drop([i for i in self.Y.columns if i not in ["Household final consumption expenditure",
                                                  "Non-profit institutions serving households' final consumption expenditure",
                                                  "Governments final consumption expenditure",
                                                  "Gross fixed capital formation",
                                                  "Changes in inventories",
                                                  "International exports"]], axis=1, inplace=True)

        self.SY.loc[:, "Household final consumption expenditure"] = self.SY.loc[:, [i for i in self.SY.columns if
                                                                          re.search(r'^PEC\w*\d', i[0])]].sum(axis=1)
        self.SY.loc[:, "Non-profit institutions serving households' final consumption expenditure"] = self.SY.loc[:,
                                                                                                 [i for i in self.SY.columns
                                                                                                  if re.search(
                                                                                                     r'^CEN\w*\d',
                                                                                                     i[0])]].sum(axis=1)
        self.SY.loc[:, "Governments final consumption expenditure"] = self.SY.loc[:, [i for i in self.SY.columns if
                                                                            re.search(r'^CEG\w*\d', i[0])]].sum(axis=1)
        self.SY.loc[:, "Gross fixed capital formation"] = self.SY.loc[:, [i for i in self.SY.columns if
                                                                re.search(r'^CO\w*\d|^ME\w*\d|^IP\w*\d', i[0])]].sum(
            axis=1)
        self.SY.loc[:, "Changes in inventories"] = self.SY.loc[:, [i for i in self.SY.columns if re.search(r'^INV', i[0])]].sum(axis=1)
        self.SY.loc[:, "International exports"] = self.SY.loc[:, [i for i in self.SY.columns if re.search(r'^INT', i[0])]].sum(axis=1)
        self.SY.drop([i for i in self.SY.columns if i not in ["Household final consumption expenditure",
                                                    "Non-profit institutions serving households' final consumption expenditure",
                                                    "Governments final consumption expenditure",
                                                    "Gross fixed capital formation",
                                                    "Changes in inventories",
                                                    "International exports"]], axis=1, inplace=True)

    def remove_codes(self):
        """
        Removes the codes from the index to only leave the name.
        :return: Modified summetric dataframes
        """
        for df in [self.A, self.Z, self.S, self.Y, self.SY]:
            df.index = pd.MultiIndex.from_tuples(df.index)
            df.index = df.index.droplevel(0)
        for df in [self.A, self.Z, self.S]:
            df.columns = pd.MultiIndex.from_tuples(df.columns)
            df.columns = df.columns.droplevel(0)

    def produce_environmental_extensions(self):
        """
        Produces environmental extensions for the symmetric OI tables produced previously
        :return: the environmental extensions satellite accounts
        """
        # Tab name changes with selected year, so identify it using "INRP-NPRI"
        emissions = self.NPRI[[i for i in self.NPRI.keys() if "INRP-NPRI" in i][0]]
        emissions.columns = list(zip(emissions.iloc[0].ffill().tolist(), emissions.iloc[2]))
        emissions = emissions.iloc[3:]
        emissions = emissions.loc[:, [i for i in emissions.columns if
                                      (i[1] in ['NAICS 4 Code', 'CAS Number', 'Substance Name (English)', 'Units']
                                       or i[1] == 'Total' and 'Air' in i[0]
                                       or i[1] == 'Total' and 'Water' in i[0])]].fillna(0)
        emissions.columns = ['NAICS 4 Code', 'CAS Number', 'Substance Name', 'Units', 'Emissions to air',
                             'Emissions to water']
        emissions.set_index('Substance Name', inplace=True)

        temp_df = emissions.groupby(emissions.index).head(n=1)
        # separating the metadata for emissions (CAS and units)
        emission_metadata = pd.DataFrame('', index=pd.MultiIndex.from_product([temp_df.index, ['air', 'water']]),
                                         columns=['CAS Number', 'Unit'])
        for emission in temp_df.index:
            emission_metadata.loc[emission, 'CAS Number'] = temp_df.loc[emission, 'CAS Number']
            emission_metadata.loc[emission, 'Unit'] = temp_df.loc[emission, 'CAS Number']
        del temp_df

        self.F = pd.pivot_table(data=emissions, index=[emissions.index], columns=['NAICS 4 Code'],
                                aggfunc=np.sum).fillna(0)
        self.F.columns.set_levels(['air', 'water'], level=0, inplace=True)
        self.F.columns = self.F.columns.rename(['compartment', 'NAICS'])
        self.F = self.F.T.unstack('compartment').T[self.F.T.unstack('compartment').T != 0].dropna(how='all').fillna(0)
