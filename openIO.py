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
import pkg_resources


class IOTables:
    def __init__(self, supply_use_excel_path, NPRI_excel, classification, aggregate_final_demand=True):
        """

        :param supply_use_excel_path: the path to the SUT excel file (e.g. /../Detail level/CA_SUT_C2016_D.xlsx)
        :param NPRI_excel: the path to the NPRI excel file (e.g. /../2017_INRP-NPRI.xlsx)
        :param classification: [string] the type of classification to adopt for the symmetric IOT ("product" or "industry")
        :param assumption: [string] the assumption used to create the symmetric IO tables ("industry technology" or
                            "fixed industry sales structure")
        :param aggregate_final_demand: [boolean] aggregating the final demand to 6 elements or not

        """
        self.SU_tables = pd.read_excel(supply_use_excel_path, None)
        self.level_of_detail = self.SU_tables['Supply'].iloc[5, 0]
        self.NPRI = pd.read_excel(NPRI_excel, None)
        self.classification = classification

        if self.classification == "product":
            self.assumption = 'industry technology'
        elif self.classification == "industry":
            self.assumption = 'fixed industry sales structure'

        self.V = pd.DataFrame()
        self.U = pd.DataFrame()
        self.A = pd.DataFrame()
        self.Z = pd.DataFrame()
        self.W = pd.DataFrame()
        self.R = pd.DataFrame()
        self.Y = pd.DataFrame()
        self.WY = pd.DataFrame()
        self.g = pd.DataFrame()
        self.inv_g = pd.DataFrame()
        self.q = pd.DataFrame()
        self.inv_q = pd.DataFrame()
        self.F = pd.DataFrame()
        self.S = pd.DataFrame()

        self.emission_metadata = pd.DataFrame()
        self.industries = []
        self.commodities = []

        self.format_tables()
        self.gimme_symmetric_iot()
        if aggregate_final_demand:
            self.aggregate_final_demand()
        self.remove_codes()
        self.extract_environmental_data()
        self.match_environmental_data_to_iots()

    def format_tables(self):
        """
        Extracts the relevant dataframes from the Excel file
        :return: the relevant dataframes
        """

        Supply_table = self.SU_tables['Supply'].copy()
        Use_table = self.SU_tables['Use_Basic'].copy()

        for i in range(0, len(Supply_table.columns)):
            if Supply_table.iloc[11, i] == 'Total':
                break
            if Supply_table.iloc[11, i] not in [np.nan, 'Industries']:
                # tuple with code + name (need code to deal with duplicate names in detailed levels)
                self.industries.append((Supply_table.iloc[12, i], Supply_table.iloc[11, i]))

        factors_of_production = []
        for i, element in enumerate(Supply_table.iloc[:, 0].tolist()):
            if type(element) == str:
                # identify by their codes
                if re.search(r'^[M,F,N,G,I,E]\w*\d', element):
                    self.commodities.append((element, Supply_table.iloc[i, 1]))
                elif re.search(r'^P\w*\d', element) or re.search(r'^GVA', element):
                    factors_of_production.append((element, Supply_table.iloc[i, 1]))

        final_demand = []
        for i in range(0, len(Use_table.columns)):
            if Use_table.iloc[11, i] == 'Total use':
                break
            if Use_table.iloc[11, i] not in [np.nan, 'Industries']:
                final_demand.append((Use_table.iloc[12, i], Use_table.iloc[11, i]))
        final_demand = [i for i in final_demand if i not in self.industries and i[1] != 'Total']

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

        # tables from M$ to $
        Supply_table *= 1000000
        Use_table *= 1000000

        # check calculated totals matched displayed totals
        assert np.allclose(Use_table.iloc[:, Use_table.columns.get_loc(('TOTAL', 'Total'))],
                           Use_table.iloc[:, :Use_table.columns.get_loc(('TOTAL', 'Total'))].sum(axis=1))
        assert np.allclose(Supply_table.iloc[Supply_table.index.get_loc(('TOTAL', 'Total'))],
                           Supply_table.iloc[:Supply_table.index.get_loc(('TOTAL', 'Total'))].sum())

        self.W = Use_table.loc[factors_of_production, self.industries]
        self.W.drop(('GVA', 'Gross value-added at basic prices'), inplace=True)
        self.Y = Use_table.loc[self.commodities, final_demand]
        self.WY = Use_table.loc[factors_of_production, final_demand]
        self.WY.drop(('GVA', 'Gross value-added at basic prices'), inplace=True)
        self.g = Supply_table.loc[[('TOTAL', 'Total')], self.industries]
        self.q = Supply_table.loc[self.commodities, [('TOTAL', 'Total')]]
        self.V = Supply_table.loc[self.commodities, self.industries]
        self.U = Use_table.loc[self.commodities, self.industries]

    def gimme_symmetric_iot(self):
        """
        Transforms Supply and Use Tables to symmetric IO tables
        :return: A, Z and S, symmetric IO tables
        """
        self.inv_q = pd.DataFrame(np.diag((1 / self.q.iloc[:, 0]).replace(np.inf, 0)), self.q.index, self.q.index)
        self.inv_g = pd.DataFrame(np.diag((1 / self.g.iloc[0]).replace(np.inf, 0)), self.g.columns, self.g.columns)

        if self.assumption == "industry technology" and self.classification == "product":
            self.Z = self.U.dot(self.inv_g.dot(self.V.T))
            self.A = self.U.dot(self.inv_g.dot(self.V.T)).dot(self.inv_q)
            self.R = self.W.dot(self.inv_g.dot(self.V.T)).dot(self.inv_q)
        elif self.assumption == "fixed industry sales structure" and self.classification == "industry":
            self.Z = self.V.T.dot(self.inv_q).dot(self.U)
            self.A = self.V.T.dot(self.inv_q).dot(self.U).dot(self.inv_g)
            self.R = self.W.dot(self.inv_g)

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

        self.WY.loc[:, "Household final consumption expenditure"] = self.WY.loc[:, [i for i in self.WY.columns if
                                                                          re.search(r'^PEC\w*\d', i[0])]].sum(axis=1)
        self.WY.loc[:, "Non-profit institutions serving households' final consumption expenditure"] = self.WY.loc[:,
                                                                                                 [i for i in self.WY.columns
                                                                                                  if re.search(
                                                                                                     r'^CEN\w*\d',
                                                                                                     i[0])]].sum(axis=1)
        self.WY.loc[:, "Governments final consumption expenditure"] = self.WY.loc[:, [i for i in self.WY.columns if
                                                                            re.search(r'^CEG\w*\d', i[0])]].sum(axis=1)
        self.WY.loc[:, "Gross fixed capital formation"] = self.WY.loc[:, [i for i in self.WY.columns if
                                                                re.search(r'^CO\w*\d|^ME\w*\d|^IP\w*\d', i[0])]].sum(
            axis=1)
        self.WY.loc[:, "Changes in inventories"] = self.WY.loc[:, [i for i in self.WY.columns if re.search(r'^INV', i[0])]].sum(axis=1)
        self.WY.loc[:, "International exports"] = self.WY.loc[:, [i for i in self.WY.columns if re.search(r'^INT', i[0])]].sum(axis=1)
        self.WY.drop([i for i in self.WY.columns if i not in ["Household final consumption expenditure",
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
        for df in [self.A, self.Z, self.W, self.R, self.Y, self.WY, self.q, self.inv_g, self.inv_q, self.V, self.U]:
            df.index = pd.MultiIndex.from_tuples(df.index)
            df.index = df.index.droplevel(0)
        for df in [self.A, self.Z, self.W, self.R, self.g, self.inv_g, self.inv_q, self.V, self.U]:
            df.columns = pd.MultiIndex.from_tuples(df.columns)
            df.columns = df.columns.droplevel(0)

    def extract_environmental_data(self):
        """
        Extracts the data from the NPRI file
        :return: the environmental extensions satellite accounts
        """
        # Tab name changes with selected year, so identify it using "INRP-NPRI"
        emissions = self.NPRI[[i for i in self.NPRI.keys() if "INRP-NPRI" in i][0]]
        emissions.columns = list(zip(emissions.iloc[0].ffill().tolist(), emissions.iloc[2]))
        emissions = emissions.iloc[3:]
        emissions = emissions.loc[:, [i for i in emissions.columns if
                                      (i[1] in ['NAICS 6 Code', 'CAS Number', 'Substance Name (English)', 'Units']
                                       or i[1] == 'Total' and 'Air' in i[0]
                                       or i[1] == 'Total' and 'Water' in i[0])]].fillna(0)
        emissions.columns = ['NAICS 6 Code', 'CAS Number', 'Substance Name', 'Units', 'Emissions to air',
                             'Emissions to water']
        emissions.set_index('Substance Name', inplace=True)

        temp_df = emissions.groupby(emissions.index).head(n=1)
        # separating the metadata for emissions (CAS and units)
        self.emission_metadata = pd.DataFrame('', index=pd.MultiIndex.from_product([temp_df.index, ['air', 'water']]),
                                         columns=['CAS Number', 'Unit'])
        for emission in temp_df.index:
            self.emission_metadata.loc[emission, 'CAS Number'] = temp_df.loc[emission, 'CAS Number']
            self.emission_metadata.loc[emission, 'Unit'] = temp_df.loc[emission, 'CAS Number']
        del temp_df

        self.F = pd.pivot_table(data=emissions, index=[emissions.index], columns=['NAICS 6 Code'],
                                aggfunc=np.sum).fillna(0)
        self.F.columns.set_levels(['air', 'water'], level=0, inplace=True)
        self.F.columns = self.F.columns.rename(['compartment', 'NAICS'])
        self.F = self.F.T.unstack('compartment').T[self.F.T.unstack('compartment').T != 0].fillna(0)

    def match_environmental_data_to_iots(self):
        """
        Links raw environmental data to the symmetric IO tables
        :return: self.F and self.S (environmental extensions)
        """
        total_emissions_origin = self.F.sum().sum()

        concordance_table = pd.read_excel(pkg_resources.resource_stream(__name__, '/Data/NAICS-IOIC.xlsx'))
        concordance_table.set_index('NAICS6', inplace=True)

        new_columns = []
        for i in self.F.columns:
            if int(i) in concordance_table.index:
                new_columns.append(concordance_table.loc[int(i), 'IOIC'])
            else:
                print('Problem ' + str(i))
        self.F.columns = new_columns
        self.F = self.F.groupby(self.F.columns, axis=1).sum()

        # adding the codes that are missing from NPRI but are in Stat Can
        IOIC_codes = [i[0] for i in self.industries]
        self.F = self.F.join(pd.DataFrame(0, index=self.F.index, columns=[i for i in IOIC_codes if
                                                                          i not in self.F.columns]))
        if self.level_of_detail == 'Detail level':
            # Animal production and aquaculture split with economic allocation
            self.F.loc[:, 'BS112A00'] = self.F.loc[:, 'BS112000'] * 0.95
            self.F.loc[:, 'BS112500'] = self.F.loc[:, 'BS112000'] * 0.05
        elif self.level_of_detail == 'Summary level':
            self.F.loc[:, 'BS11A'] = self.F.loc[:, ['BS111A00', 'BS112000']].sum(axis=1)
            self.F.loc[:, 'BS113'] = self.F.loc[:, 'BS113000']
            self.F.loc[:, 'BS114'] = self.F.loc[:, 'BS1114A0']
            self.F.loc[:, 'BS115'] = self.F.loc[:, 'BS115A00']
            self.F.loc[:, 'BS210'] = self.F.loc[:, [i for i in self.F.columns if re.search(r'^BS21', i)]].sum(axis=1)
            self.F.loc[:, 'BS220'] = self.F.loc[:, [i for i in self.F.columns if re.search(r'^BS22', i)]].sum(axis=1)
            self.F.loc[:, 'BS23A'] = self.F.loc[:, 'BS23A000']
            self.F.loc[:, 'BS3A0'] = self.F.loc[:, [i for i in self.F.columns if re.search(r'^BS3', i)]].sum(axis=1)
            self.F.loc[:, 'BS410'] = self.F.loc[:, [i for i in self.F.columns if re.search(r'^BS41', i)]].sum(axis=1)
            self.F.loc[:, 'BS4B0'] = self.F.loc[:, [i for i in self.F.columns if re.search(r'^BS48|^BS49', i)]].sum(axis=1)
            self.F.loc[:, 'BS510'] = self.F.loc[:, 'BS518000']
            self.F.loc[:, 'BS5B0'] = self.F.loc[:, 'BS531100']
            self.F.loc[:, 'BS540'] = self.F.loc[:, [i for i in self.F.columns if re.search(r'^BS54', i)]].sum(axis=1)
            self.F.loc[:, 'BS560'] = self.F.loc[:, [i for i in self.F.columns if re.search(r'^BS56', i)]].sum(axis=1)
            self.F.loc[:, 'BS610'] = self.F.loc[:, [i for i in self.F.columns if re.search(r'^BS61', i)]].sum(axis=1)
            self.F.loc[:, 'BS720'] = self.F.loc[:, 'BS721A00']
            self.F.loc[:, 'BS810'] = self.F.loc[:, [i for i in self.F.columns if re.search(r'^BS81', i)]].sum(axis=1)
            self.F.loc[:, 'GS610'] = self.F.loc[:, [i for i in self.F.columns if re.search(r'^GS61', i)]].sum(axis=1)
            self.F.loc[:, 'GS620'] = self.F.loc[:, 'GS622000']
            self.F.loc[:, 'GS911'] = self.F.loc[:, [i for i in self.F.columns if re.search(r'^GS911', i)]].sum(axis=1)
            self.F.loc[:, 'GS913'] = self.F.loc[:, 'GS913000']
        elif self.level_of_detail == 'Link-1997 level':
            self.F.loc[:, 'BS211000'] += self.F.loc[:, 'BS211110']
            self.F.loc[:, 'BS211000'] += self.F.loc[:, 'BS211140']
            self.F.loc[:, 'BS213000'] += self.F.loc[:, 'BS21311A']
            self.F.loc[:, 'BS324000'] += self.F.loc[:, 'BS324110']
            self.F.loc[:, 'BS324000'] += self.F.loc[:, 'BS3241A0']
            self.F.loc[:, 'BS325B00'] += self.F.loc[:, 'BS325200']
            self.F.loc[:, 'BS325B00'] += self.F.loc[:, 'BS325500']
            self.F.loc[:, 'BS333A00'] += self.F.loc[:, 'BS333200']
            self.F.loc[:, 'BS333A00'] += self.F.loc[:, 'BS333300']
            self.F.loc[:, 'BS336100'] += self.F.loc[:, 'BS336110']
            self.F.loc[:, 'BS336100'] += self.F.loc[:, 'BS336120']
            self.F.loc[:, 'BS336300'] += self.F.loc[:, [i for i in self.F.columns if
                                                        i not in IOIC_codes and re.search(r'^BS3363',i)]].sum(axis=1)
            self.F.loc[:, 'BS410000'] += self.F.loc[:, [i for i in self.F.columns if
                                                        i not in IOIC_codes and re.search(r'^BS41',i)]].sum(axis=1)
            self.F.loc[:, 'BS541B00'] += self.F.loc[:, 'BS541700']
            self.F.loc[:, 'BS541B00'] += self.F.loc[:, 'BS541900']
            self.F.loc[:, 'BS561B00'] += self.F.loc[:, 'BS561A00']
        elif self.level_of_detail == 'Link-1961 level':
            self.F.loc[:, 'BS11B00'] += self.F.loc[:, 'BS111A00']
            self.F.loc[:, 'BS11B00'] += self.F.loc[:, 'BS112000']
            self.F.loc[:, 'BS11300'] += self.F.loc[:, 'BS113000']
            self.F.loc[:, 'BS11400'] += self.F.loc[:, 'BS1114A0']
            self.F.loc[:, 'BS11500'] += self.F.loc[:, 'BS115A00']
            self.F.loc[:, 'BS21100'] += self.F.loc[:, [i for i in self.F.columns if re.search(r'^BS211', i)]].sum(axis=1)
            self.F.loc[:, 'BS21210'] += self.F.loc[:, [i for i in self.F.columns if re.search(r'^BS212', i)]].sum(axis=1)
            self.F.loc[:, 'BS21300'] += self.F.loc[:, 'BS21311A']
            self.F.loc[:, 'BS22110'] += self.F.loc[:, 'BS221100']
            self.F.loc[:, 'BS221A0'] += self.F.loc[:, 'BS221200']
            self.F.loc[:, 'BS221A0'] += self.F.loc[:, 'BS221300']
            self.F.loc[:, 'BS23A00'] += self.F.loc[:, 'BS23A000']
            self.F.loc[:, 'BS31110'] += self.F.loc[:, 'BS311100']
            self.F.loc[:, 'BS31130'] += self.F.loc[:, 'BS311300']
            self.F.loc[:, 'BS31140'] += self.F.loc[:, 'BS311400']
            self.F.loc[:, 'BS31150'] += self.F.loc[:, 'BS311500']
            self.F.loc[:, 'BS31160'] += self.F.loc[:, 'BS311600']
            self.F.loc[:, 'BS31170'] += self.F.loc[:, 'BS311700']
            self.F.loc[:, 'BS311A0'] += self.F.loc[:, 'BS311200']
            self.F.loc[:, 'BS311A0'] += self.F.loc[:, 'BS311800']
            self.F.loc[:, 'BS311A0'] += self.F.loc[:, 'BS311900']
            self.F.loc[:, 'BS31211'] += self.F.loc[:, 'BS312110']
            self.F.loc[:, 'BS31212'] += self.F.loc[:, 'BS312120']
            self.F.loc[:, 'BS3121A'] += self.F.loc[:, 'BS3121A0']
            self.F.loc[:, 'BS31220'] += self.F.loc[:, 'BS312200']
            self.F.loc[:, 'BS31A00'] += self.F.loc[:, 'BS31A000']
            self.F.loc[:, 'BS31B00'] += self.F.loc[:, 'BS31B000']
            self.F.loc[:, 'BS32100'] += self.F.loc[:, [i for i in self.F.columns if re.search(r'^BS321', i)]].sum(axis=1)
            self.F.loc[:, 'BS32210'] += self.F.loc[:, 'BS322100']
            self.F.loc[:, 'BS32220'] += self.F.loc[:, 'BS322200']
            self.F.loc[:, 'BS32300'] += self.F.loc[:, 'BS323000']
            self.F.loc[:, 'BS32400'] += self.F.loc[:, [i for i in self.F.columns if re.search(r'^BS324', i)]].sum(axis=1)
            self.F.loc[:, 'BS32510'] += self.F.loc[:, 'BS325100']
            self.F.loc[:, 'BS32530'] += self.F.loc[:, 'BS325300']
            self.F.loc[:, 'BS32540'] += self.F.loc[:, 'BS325400']
            self.F.loc[:, 'BS325C0'] += self.F.loc[:, 'BS325200']
            self.F.loc[:, 'BS325C0'] += self.F.loc[:, 'BS325500']
            self.F.loc[:, 'BS325C0'] += self.F.loc[:, 'BS325600']
            self.F.loc[:, 'BS325C0'] += self.F.loc[:, 'BS325900']
            self.F.loc[:, 'BS32610'] += self.F.loc[:, 'BS326100']
            self.F.loc[:, 'BS32620'] += self.F.loc[:, 'BS326200']
            self.F.loc[:, 'BS327A0'] += self.F.loc[:, 'BS327A00']
            self.F.loc[:, 'BS32730'] += self.F.loc[:, 'BS327300']
            self.F.loc[:, 'BS33100'] += self.F.loc[:, [i for i in self.F.columns if re.search(r'^BS331', i)]].sum(axis=1)
            self.F.loc[:, 'BS33200'] += self.F.loc[:, [i for i in self.F.columns if re.search(r'^BS332', i)]].sum(axis=1)
            self.F.loc[:, 'BS33300'] += self.F.loc[:, [i for i in self.F.columns if re.search(r'^BS333', i)]].sum(axis=1)
            self.F.loc[:, 'BS33410'] += self.F.loc[:, 'BS334100']
            self.F.loc[:, 'BS334B0'] += self.F.loc[:, 'BS334200']
            self.F.loc[:, 'BS334B0'] += self.F.loc[:, 'BS334400']
            self.F.loc[:, 'BS334B0'] += self.F.loc[:, 'BS334A00']
            self.F.loc[:, 'BS335A0'] += self.F.loc[:, 'BS335100']
            self.F.loc[:, 'BS335A0'] += self.F.loc[:, 'BS335300']
            self.F.loc[:, 'BS335A0'] += self.F.loc[:, 'BS335900']
            self.F.loc[:, 'BS33520'] += self.F.loc[:, 'BS335200']
            self.F.loc[:, 'BS33610'] += self.F.loc[:, [i for i in self.F.columns if re.search(r'^BS3361', i)]].sum(axis=1)
            self.F.loc[:, 'BS33620'] += self.F.loc[:, 'BS336200']
            self.F.loc[:, 'BS33630'] += self.F.loc[:, [i for i in self.F.columns if re.search(r'^BS3363', i)]].sum(axis=1)
            self.F.loc[:, 'BS33640'] += self.F.loc[:, 'BS336400']
            self.F.loc[:, 'BS33650'] += self.F.loc[:, 'BS336500']
            self.F.loc[:, 'BS33660'] += self.F.loc[:, 'BS336600']
            self.F.loc[:, 'BS33690'] += self.F.loc[:, 'BS336900']
            self.F.loc[:, 'BS33700'] += self.F.loc[:, [i for i in self.F.columns if re.search(r'^BS337', i)]].sum(axis=1)
            self.F.loc[:, 'BS33900'] += self.F.loc[:, [i for i in self.F.columns if re.search(r'^BS339', i)]].sum(axis=1)
            self.F.loc[:, 'BS41000'] += self.F.loc[:,  [i for i in self.F.columns if re.search(r'^BS41', i)]].sum(axis=1)
            self.F.loc[:, 'BS48100'] += self.F.loc[:, 'BS481000']
            self.F.loc[:, 'BS48200'] += self.F.loc[:, 'BS482000']
            self.F.loc[:, 'BS48400'] += self.F.loc[:, 'BS484000']
            self.F.loc[:, 'BS48600'] += self.F.loc[:, [i for i in self.F.columns if re.search(r'^BS486', i)]].sum(axis=1)
            self.F.loc[:, 'BS48B00'] += self.F.loc[:, 'BS488000']
            self.F.loc[:, 'BS49300'] += self.F.loc[:, 'BS493000']
            self.F.loc[:, 'BS51B00'] += self.F.loc[:, 'BS518000']
            self.F.loc[:, 'BS53110'] += self.F.loc[:, 'BS531100']
            self.F.loc[:, 'BS541C0'] += self.F.loc[:, 'BS541300']
            self.F.loc[:, 'BS541D0'] += self.F.loc[:, 'BS541700']
            self.F.loc[:, 'BS541D0'] += self.F.loc[:, 'BS541900']
            self.F.loc[:, 'BS56100'] += self.F.loc[:, 'BS561A00']
            self.F.loc[:, 'BS56200'] += self.F.loc[:, 'BS562000']
            self.F.loc[:, 'BS61000'] += self.F.loc[:, 'BS610000']
            self.F.loc[:, 'BS72000'] += self.F.loc[:, 'BS721A00']
            self.F.loc[:, 'BS81100'] += self.F.loc[:, [i for i in self.F.columns if re.search(r'^BS811', i)]].sum(axis=1)
            self.F.loc[:, 'BS81A00'] += self.F.loc[:, [i for i in self.F.columns if re.search(r'^BS812', i)]].sum(axis=1)
            self.F.loc[:, 'GS61130'] += self.F.loc[:, 'GS611300']
            self.F.loc[:, 'GS611B0'] += self.F.loc[:, 'GS611200']
            self.F.loc[:, 'GS62200'] += self.F.loc[:, 'GS622000']
            self.F.loc[:, 'GS91100'] += self.F.loc[:, [i for i in self.F.columns if re.search(r'^BS911', i)]].sum(axis=1)
            self.F.loc[:, 'GS91300'] += self.F.loc[:, 'GS913000']

        self.F = self.F.reindex(IOIC_codes, axis=1)
        # check the order is the same before replacing codes with names
        assert all(self.F.columns == [i[0] for i in self.industries])
        self.F.columns = [i[1] for i in self.industries]

        if self.classification == 'industry':
            self.S = self.F.dot(self.inv_g)

        if self.classification == 'product':
            self.F = self.F.dot(self.V.dot(self.inv_g).T)
            self.S = self.F.dot(self.inv_q)

        # assert that most of the emissions (>99%) given by the NPRI are present in self.F
        assert self.F.sum().sum() / total_emissions_origin > 0.99
        assert self.F.sum().sum() / total_emissions_origin < 1.01

