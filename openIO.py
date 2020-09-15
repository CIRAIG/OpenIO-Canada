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
    def __init__(self, supply_use_excel_path, classification):
        """
        :param supply_use_excel_path: the path to the SUT excel file (e.g. /../Detail level/CA_SUT_C2016_D.xlsx)
        :param classification: [string] the type of classification to adopt for the symmetric IOT ("product" or "industry")
        """

        self.SU_tables = pd.read_excel(supply_use_excel_path, None)
        self.level_of_detail = self.SU_tables['Supply'].iloc[5, 0]
        self.NPRI = pd.read_excel((pkg_resources.resource_stream(__name__, '/Data/2017_INRP-NPRI.xlsx')), None)
        self.classification = classification

        if self.classification == "product":
            self.assumption = 'industry technology'
        elif self.classification == "industry":
            self.assumption = 'fixed industry sales structure'

        # values
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
        self.FY = pd.DataFrame()
        self.C = pd.DataFrame()

        # metadata
        self.emission_metadata = pd.DataFrame()
        self.methods_metadata = pd.DataFrame()
        self.industries = []
        self.commodities = []

        # methods executed
        self.format_tables()
        self.gimme_symmetric_iot()
        self.aggregate_final_demand()
        self.remove_codes()
        self.extract_environmental_data()
        self.match_environmental_data_to_iots()
        self.characterization_matrix()

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
                                       or i[1] == 'Total' and 'Water' in i[0]
                                       or i[1] == 'Total' and 'Land' in i[0])]].fillna(0)
        emissions.columns = ['NAICS 6 Code', 'CAS Number', 'Substance Name', 'Units', 'Emissions to air',
                             'Emissions to water', 'Emissions to land']
        emissions.set_index('Substance Name', inplace=True)

        temp_df = emissions.groupby(emissions.index).head(n=1)
        # separating the metadata for emissions (CAS and units)
        self.emission_metadata = pd.DataFrame('', index=pd.MultiIndex.from_product([temp_df.index,
                                                                                    ['Air', 'Water', 'Soil']]),
                                              columns=['CAS Number', 'Unit'])
        for emission in temp_df.index:
            self.emission_metadata.loc[emission, 'CAS Number'] = temp_df.loc[emission, 'CAS Number']
            self.emission_metadata.loc[emission, 'Unit'] = temp_df.loc[emission, 'Units']
        del temp_df

        self.F = pd.pivot_table(data=emissions, index=[emissions.index], columns=['NAICS 6 Code'],
                                aggfunc=np.sum).fillna(0)
        self.F.columns.set_levels(['Air', 'Water', 'Soil'], level=0, inplace=True)
        self.F.columns = self.F.columns.rename(['compartment', 'NAICS'])
        self.F = self.F.T.unstack('compartment').T[self.F.T.unstack('compartment').T != 0].fillna(0)

        # convert all tonnes and grams to kgs and change in metadata as well
        self.F.loc[
            [i for i in self.emission_metadata.index if self.emission_metadata.loc[i, 'Unit'] == 'tonnes']] *= 1000
        self.emission_metadata.loc[[i for i in self.emission_metadata.index if
                                    self.emission_metadata.loc[i, 'Unit'] == 'tonnes'], 'Unit'] = 'kg'
        self.F.loc[
            [i for i in self.emission_metadata.index if self.emission_metadata.loc[i, 'Unit'] == 'grams']] /= 1000
        self.emission_metadata.loc[[i for i in self.emission_metadata.index if
                                    self.emission_metadata.loc[i, 'Unit'] == 'grams'], 'Unit'] = 'kg'

    def match_environmental_data_to_iots(self):
        """
        Links raw environmental data to the symmetric IO tables
        :return: self.F and self.S (environmental extensions)
        """

        # ---------------------POLLUTANTS-------------------------

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

        # assert that most of the emissions (>98%) given by the NPRI are present in self.F
        assert self.F.sum().sum() / total_emissions_origin > 0.98
        assert self.F.sum().sum() / total_emissions_origin < 1.02

        GHGs = extract_data_from_csv('GHG_emissions.csv')
        NRG = extract_data_from_csv('Energy_use.csv')
        water = extract_data_from_csv('Water_use.csv')

        # households emissions
        self.FY = pd.DataFrame([[
            GHGs.loc[[i for i in GHGs.index if 'Households' in i]].sum().loc['VALUE']*1000000, 0, 0, 0, 0, 0],
            [NRG.loc[[i for i in NRG.index if 'Households' in i]].sum().loc['VALUE'], 0, 0, 0, 0, 0],
            [water.loc[[i for i in water.index if 'Households' in i]].sum().loc['VALUE'], 0, 0, 0, 0, 0]],
            index=[('GHGs',''), ('Energy use',''), ('Water use','')],
            columns=self.Y.columns)
        self.FY = pd.concat([pd.DataFrame(0, self.F.index, self.Y.columns),self.FY])

        industries_nrg = select_industries_emissions(NRG)
        industries_water = select_industries_emissions(water)
        industries_ghgs = select_industries_emissions(GHGs)

        # ---------------------GHGs-------------------------

        if self.level_of_detail == 'Link-1961 level':
            industries_ghgs = industries_ghgs.drop([i for i in industries_ghgs.index if i not in IOIC_codes])
        elif self.level_of_detail == 'Link-1997 level':
            key_changes = dict.fromkeys(industries_ghgs.index, '')
            for key in key_changes:
                if key + '0' in IOIC_codes:
                    key_changes[key] = key + '0'

            # economic allocations. Be my guest if you wanna verify :)
            industries_ghgs.loc['BS111A00'] = 0.55 * industries_ghgs.loc['BS11B00']
            industries_ghgs.loc['BS1114A0'] = 0.05 * industries_ghgs.loc['BS11B00']
            industries_ghgs.loc['BS112000'] = 0.40 * industries_ghgs.loc['BS11B00']
            industries_ghgs.loc['BS115A00'] = 0.39 * industries_ghgs.loc['BS11500']
            industries_ghgs.loc['BS115300'] = 0.61 * industries_ghgs.loc['BS11500']
            industries_ghgs.loc['BS212210'] = 0.15 * industries_ghgs.loc['BS21220']
            industries_ghgs.loc['BS212220'] = 0.35 * industries_ghgs.loc['BS21220']
            industries_ghgs.loc['BS212230'] = 0.4 * industries_ghgs.loc['BS21220']
            industries_ghgs.loc['BS212290'] = 0.1 * industries_ghgs.loc['BS21220']
            industries_ghgs.loc['BS212310'] = 0.16 * industries_ghgs.loc['BS21230']
            industries_ghgs.loc['BS212320'] = 0.15 * industries_ghgs.loc['BS21230']
            industries_ghgs.loc['BS212392'] = 0.18 * industries_ghgs.loc['BS21230']
            industries_ghgs.loc['BS21239A'] = 0.13 * industries_ghgs.loc['BS21230']
            industries_ghgs.loc['BS212396'] = 0.38 * industries_ghgs.loc['BS21230']
            industries_ghgs.loc['BS221200'] = 0.9 * industries_ghgs.loc['BS221A0']
            industries_ghgs.loc['BS221300'] = 0.1 * industries_ghgs.loc['BS221A0']
            industries_ghgs.loc['BS311200'] = 0.35 * industries_ghgs.loc['BS311A0']
            industries_ghgs.loc['BS311800'] = 0.31 * industries_ghgs.loc['BS311A0']
            industries_ghgs.loc['BS311900'] = 0.33 * industries_ghgs.loc['BS311A0']
            industries_ghgs.loc['BS321100'] = 0.52 * industries_ghgs.loc['BS32100']
            industries_ghgs.loc['BS321200'] = 0.22 * industries_ghgs.loc['BS32100']
            industries_ghgs.loc['BS321900'] = 0.26 * industries_ghgs.loc['BS32100']
            industries_ghgs.loc['BS325B00'] = 0.57 * industries_ghgs.loc['BS325C0']
            industries_ghgs.loc['BS325600'] = 0.20 * industries_ghgs.loc['BS325C0']
            industries_ghgs.loc['BS325900'] = 0.23 * industries_ghgs.loc['BS325C0']
            industries_ghgs.loc['BS331100'] = 0.17 * industries_ghgs.loc['BS33100']
            industries_ghgs.loc['BS331200'] = 0.07 * industries_ghgs.loc['BS33100']
            industries_ghgs.loc['BS331300'] = 0.18 * industries_ghgs.loc['BS33100']
            industries_ghgs.loc['BS331400'] = 0.54 * industries_ghgs.loc['BS33100']
            industries_ghgs.loc['BS331500'] = 0.04 * industries_ghgs.loc['BS33100']
            industries_ghgs.loc['BS332100'] = 0.04 * industries_ghgs.loc['BS33200']
            industries_ghgs.loc['BS332A00'] = 0.16 * industries_ghgs.loc['BS33200']
            industries_ghgs.loc['BS332300'] = 0.41 * industries_ghgs.loc['BS33200']
            industries_ghgs.loc['BS332400'] = 0.1 * industries_ghgs.loc['BS33200']
            industries_ghgs.loc['BS332500'] = 0.05 * industries_ghgs.loc['BS33200']
            industries_ghgs.loc['BS332600'] = 0.03 * industries_ghgs.loc['BS33200']
            industries_ghgs.loc['BS332700'] = 0.16 * industries_ghgs.loc['BS33200']
            industries_ghgs.loc['BS332800'] = 0.06 * industries_ghgs.loc['BS33200']
            industries_ghgs.loc['BS333100'] = 0.25 * industries_ghgs.loc['BS33300']
            industries_ghgs.loc['BS333A00'] = 0.26 * industries_ghgs.loc['BS33300']
            industries_ghgs.loc['BS333400'] = 0.1 * industries_ghgs.loc['BS33300']
            industries_ghgs.loc['BS333500'] = 0.12 * industries_ghgs.loc['BS33300']
            industries_ghgs.loc['BS333600'] = 0.04 * industries_ghgs.loc['BS33300']
            industries_ghgs.loc['BS333900'] = 0.22 * industries_ghgs.loc['BS33300']
            industries_ghgs.loc['BS334200'] = 0.25 * industries_ghgs.loc['BS334B0']
            industries_ghgs.loc['BS334A00'] = 0.46 * industries_ghgs.loc['BS334B0']
            industries_ghgs.loc['BS334400'] = 0.29 * industries_ghgs.loc['BS334B0']
            industries_ghgs.loc['BS335100'] = 0.12 * industries_ghgs.loc['BS335A0']
            industries_ghgs.loc['BS335300'] = 0.49 * industries_ghgs.loc['BS335A0']
            industries_ghgs.loc['BS335900'] = 0.39 * industries_ghgs.loc['BS335A0']
            industries_ghgs.loc['BS337100'] = 0.54 * industries_ghgs.loc['BS33700']
            industries_ghgs.loc['BS337200'] = 0.36 * industries_ghgs.loc['BS33700']
            industries_ghgs.loc['BS337900'] = 0.1 * industries_ghgs.loc['BS33700']
            industries_ghgs.loc['BS339100'] = 0.28 * industries_ghgs.loc['BS33900']
            industries_ghgs.loc['BS339900'] = 0.72 * industries_ghgs.loc['BS33900']
            industries_ghgs.loc['BS486A00'] = 0.51 * industries_ghgs.loc['BS48600']
            industries_ghgs.loc['BS486200'] = 0.49 * industries_ghgs.loc['BS48600']
            industries_ghgs.loc['BS485100'] = 0.11 * industries_ghgs.loc['BS48B00']
            industries_ghgs.loc['BS48A000'] = 0.09 * industries_ghgs.loc['BS48B00']
            industries_ghgs.loc['BS485300'] = 0.06 * industries_ghgs.loc['BS48B00']
            industries_ghgs.loc['BS488000'] = 0.75 * industries_ghgs.loc['BS48B00']
            industries_ghgs.loc['BS5121A0'] = 0.77 * industries_ghgs.loc['BS51200']
            industries_ghgs.loc['BS512130'] = 0.14 * industries_ghgs.loc['BS51200']
            industries_ghgs.loc['BS512200'] = 0.08 * industries_ghgs.loc['BS51200']
            industries_ghgs.loc['BS511100'] = 0.09 * industries_ghgs.loc['BS51B00']
            industries_ghgs.loc['BS511200'] = 0.11 * industries_ghgs.loc['BS51B00']
            industries_ghgs.loc['BS51A000'] = 0.74 * industries_ghgs.loc['BS51B00']
            industries_ghgs.loc['BS518000'] = 0.06 * industries_ghgs.loc['BS51B00']
            industries_ghgs.loc['BS521000'] = 0.002 * industries_ghgs.loc['BS52B00']
            industries_ghgs.loc['BS5221A0'] = 0.469 * industries_ghgs.loc['BS52B00']
            industries_ghgs.loc['BS522130'] = 0.042 * industries_ghgs.loc['BS52B00']
            industries_ghgs.loc['BS522A00'] = 0.099 * industries_ghgs.loc['BS52B00']
            industries_ghgs.loc['BS52A000'] = 0.306 * industries_ghgs.loc['BS52B00']
            industries_ghgs.loc['BS524200'] = 0.082 * industries_ghgs.loc['BS52B00']
            industries_ghgs.loc['BS532100'] = 0.28 * industries_ghgs.loc['BS53B00']
            industries_ghgs.loc['BS53A000'] = 0.72 * industries_ghgs.loc['BS53B00']
            industries_ghgs.loc['BS541A00'] = 0.52 * industries_ghgs.loc['BS541C0']
            industries_ghgs.loc['BS541300'] = 0.48 * industries_ghgs.loc['BS541C0']
            industries_ghgs.loc['BS541B00'] = 0.48 * industries_ghgs.loc['BS541D0']
            industries_ghgs.loc['BS541500'] = 0.52 * industries_ghgs.loc['BS541D0']
            industries_ghgs.loc['BS561B00'] = 0.61 * industries_ghgs.loc['BS56100']
            industries_ghgs.loc['BS561500'] = 0.06 * industries_ghgs.loc['BS56100']
            industries_ghgs.loc['BS561600'] = 0.09 * industries_ghgs.loc['BS56100']
            industries_ghgs.loc['BS561700'] = 0.23 * industries_ghgs.loc['BS56100']
            industries_ghgs.loc['BS531A00'] = 0.62 * industries_ghgs.loc['BS5A000']
            industries_ghgs.loc['BS551113'] = 0.38 * industries_ghgs.loc['BS5A000']
            industries_ghgs.loc['BS621100'] = 0.44 * industries_ghgs.loc['BS62000']
            industries_ghgs.loc['BS621200'] = 0.21 * industries_ghgs.loc['BS62000']
            industries_ghgs.loc['BS621A00'] = 0.17 * industries_ghgs.loc['BS62000']
            industries_ghgs.loc['BS623000'] = 0.11 * industries_ghgs.loc['BS62000']
            industries_ghgs.loc['BS624000'] = 0.07 * industries_ghgs.loc['BS62000']
            industries_ghgs.loc['BS71A000'] = 0.37 * industries_ghgs.loc['BS71000']
            industries_ghgs.loc['BS713A00'] = 0.37 * industries_ghgs.loc['BS71000']
            industries_ghgs.loc['BS713200'] = 0.26 * industries_ghgs.loc['BS71000']
            industries_ghgs.loc['BS721100'] = 0.19 * industries_ghgs.loc['BS72000']
            industries_ghgs.loc['BS721A00'] = 0.03 * industries_ghgs.loc['BS72000']
            industries_ghgs.loc['BS722000'] = 0.78 * industries_ghgs.loc['BS72000']
            industries_ghgs.loc['BS811100'] = 0.53 * industries_ghgs.loc['BS81100']
            industries_ghgs.loc['BS811A00'] = 0.47 * industries_ghgs.loc['BS81100']
            industries_ghgs.loc['BS812A00'] = 0.59 * industries_ghgs.loc['BS81A00']
            industries_ghgs.loc['BS812200'] = 0.11 * industries_ghgs.loc['BS81A00']
            industries_ghgs.loc['BS812300'] = 0.12 * industries_ghgs.loc['BS81A00']
            industries_ghgs.loc['BS814000'] = 0.18 * industries_ghgs.loc['BS81A00']
            industries_ghgs.loc['GS611100'] = 0.83 * industries_ghgs.loc['GS611B0']
            industries_ghgs.loc['GS611200'] = 0.16 * industries_ghgs.loc['GS611B0']
            industries_ghgs.loc['GS611A00'] = 0.01 * industries_ghgs.loc['GS611B0']
            industries_ghgs.loc['GS911100'] = 0.27 * industries_ghgs.loc['GS91100']
            industries_ghgs.loc['GS911A00'] = 0.73 * industries_ghgs.loc['GS91100']

            new_index = []
            for code in industries_ghgs.index:
                if code in key_changes:
                    if key_changes[code] != '':
                        new_index.append(key_changes[code])
                    else:
                        new_index.append(code)
                else:
                    new_index.append(code)

            industries_ghgs.index = new_index
            industries_ghgs = industries_ghgs.loc[IOIC_codes]
        elif self.level_of_detail == 'Summary level':
            industries_ghgs.index = [i[:-2] for i in industries_ghgs.index]
            industries_ghgs = industries_ghgs.groupby(industries_ghgs.index).sum()

            industries_ghgs.loc['BS11A'] += industries_ghgs.loc[['BS11B', 'BS111']].sum()
            industries_ghgs.loc['BS210'] = industries_ghgs.loc[['BS211', 'BS212', 'BS213']].sum()
            industries_ghgs.loc['BS220'] = industries_ghgs.loc[['BS221']].sum()
            industries_ghgs.loc['BS3A0'] = industries_ghgs.loc[
                ['BS311', 'BS312', 'BS31A', 'BS31B', 'BS321', 'BS322', 'BS323',
                 'BS324', 'BS325', 'BS326', 'BS327', 'BS331', 'BS332', 'BS333',
                 'BS334', 'BS335', 'BS336', 'BS337', 'BS339']].sum()
            industries_ghgs.loc['BS4A0'] = industries_ghgs.loc[['BS4AA', 'BS453']].sum()
            industries_ghgs.loc['BS4B0'] = industries_ghgs.loc[
                ['BS481', 'BS482', 'BS483', 'BS484', 'BS48B', 'BS486', 'BS49A', 'BS493']].sum()
            industries_ghgs.loc['BS510'] = industries_ghgs.loc[['BS512', 'BS515', 'BS51B']].sum()
            industries_ghgs.loc['BS5B0'] = industries_ghgs.loc[['BS52B', 'BS524', 'BS531', 'BS53B', 'BS5A0']].sum()
            industries_ghgs.loc['BS540'] = industries_ghgs.loc[['BS541']].sum()
            industries_ghgs.loc['BS560'] = industries_ghgs.loc[['BS561', 'BS562']].sum()
            industries_ghgs.loc['BS810'] = industries_ghgs.loc[['BS811', 'BS81A', 'BS813']].sum()
            industries_ghgs.loc['FC100'] = industries_ghgs.loc[['FC110', 'FC120', 'FC130']].sum()
            industries_ghgs.loc['NP000'] = industries_ghgs.loc[['NP610', 'NP624', 'NP710', 'NP813', 'NPA00']].sum()
            industries_ghgs.loc['GS610'] = industries_ghgs.loc[['GS611']].sum()
            industries_ghgs.loc['GS620'] = industries_ghgs.loc[['GS622', 'GS623']].sum()

            industries_ghgs = industries_ghgs.reindex(IOIC_codes).fillna(0)

        elif self.level_of_detail == 'Detail level':
            key_changes = dict.fromkeys(industries_ghgs.index, '')
            for key in key_changes:
                if key + '0' in IOIC_codes:
                    key_changes[key] = key + '0'

            # economic allocations. Be my guest if you wanna verify :)
            industries_ghgs.loc['BS111A00'] = 0.55 * industries_ghgs.loc['BS11B00']
            industries_ghgs.loc['BS1114A0'] = 0.05 * industries_ghgs.loc['BS11B00']
            industries_ghgs.loc['BS112A00'] = 0.38 * industries_ghgs.loc['BS11B00']
            industries_ghgs.loc['BS112500'] = 0.02 * industries_ghgs.loc['BS11B00']
            industries_ghgs.loc['BS115A00'] = 0.39 * industries_ghgs.loc['BS11500']
            industries_ghgs.loc['BS115300'] = 0.61 * industries_ghgs.loc['BS11500']
            industries_ghgs.loc['BS211110'] = 0.5 * industries_ghgs.loc['BS21100']
            industries_ghgs.loc['BS211140'] = 0.5 * industries_ghgs.loc['BS21100']
            industries_ghgs.loc['BS212210'] = 0.15 * industries_ghgs.loc['BS21220']
            industries_ghgs.loc['BS212220'] = 0.35 * industries_ghgs.loc['BS21220']
            industries_ghgs.loc['BS212230'] = 0.4 * industries_ghgs.loc['BS21220']
            industries_ghgs.loc['BS212290'] = 0.1 * industries_ghgs.loc['BS21220']
            industries_ghgs.loc['BS212310'] = 0.16 * industries_ghgs.loc['BS21230']
            industries_ghgs.loc['BS212320'] = 0.15 * industries_ghgs.loc['BS21230']
            industries_ghgs.loc['BS212392'] = 0.18 * industries_ghgs.loc['BS21230']
            industries_ghgs.loc['BS21239A'] = 0.13 * industries_ghgs.loc['BS21230']
            industries_ghgs.loc['BS212396'] = 0.38 * industries_ghgs.loc['BS21230']
            industries_ghgs.loc['BS21311A'] = 0.73 * industries_ghgs.loc['BS21300']
            industries_ghgs.loc['BS21311B'] = 0.27 * industries_ghgs.loc['BS21300']
            industries_ghgs.loc['BS221200'] = 0.9 * industries_ghgs.loc['BS221A0']
            industries_ghgs.loc['BS221300'] = 0.1 * industries_ghgs.loc['BS221A0']
            industries_ghgs.loc['BS311200'] = 0.35 * industries_ghgs.loc['BS311A0']
            industries_ghgs.loc['BS311800'] = 0.31 * industries_ghgs.loc['BS311A0']
            industries_ghgs.loc['BS311900'] = 0.33 * industries_ghgs.loc['BS311A0']
            industries_ghgs.loc['BS321100'] = 0.52 * industries_ghgs.loc['BS32100']
            industries_ghgs.loc['BS321200'] = 0.22 * industries_ghgs.loc['BS32100']
            industries_ghgs.loc['BS321900'] = 0.26 * industries_ghgs.loc['BS32100']
            industries_ghgs.loc['BS324110'] = 0.91 * industries_ghgs.loc['BS32400']
            industries_ghgs.loc['BS3241A0'] = 0.09 * industries_ghgs.loc['BS32400']
            industries_ghgs.loc['BS325200'] = 0.43 * industries_ghgs.loc['BS325C0']
            industries_ghgs.loc['BS325500'] = 0.13 * industries_ghgs.loc['BS325C0']
            industries_ghgs.loc['BS325600'] = 0.20 * industries_ghgs.loc['BS325C0']
            industries_ghgs.loc['BS325900'] = 0.23 * industries_ghgs.loc['BS325C0']
            industries_ghgs.loc['BS331100'] = 0.17 * industries_ghgs.loc['BS33100']
            industries_ghgs.loc['BS331200'] = 0.07 * industries_ghgs.loc['BS33100']
            industries_ghgs.loc['BS331300'] = 0.18 * industries_ghgs.loc['BS33100']
            industries_ghgs.loc['BS331400'] = 0.54 * industries_ghgs.loc['BS33100']
            industries_ghgs.loc['BS331500'] = 0.04 * industries_ghgs.loc['BS33100']
            industries_ghgs.loc['BS332100'] = 0.04 * industries_ghgs.loc['BS33200']
            industries_ghgs.loc['BS332A00'] = 0.16 * industries_ghgs.loc['BS33200']
            industries_ghgs.loc['BS332300'] = 0.41 * industries_ghgs.loc['BS33200']
            industries_ghgs.loc['BS332400'] = 0.1 * industries_ghgs.loc['BS33200']
            industries_ghgs.loc['BS332500'] = 0.05 * industries_ghgs.loc['BS33200']
            industries_ghgs.loc['BS332600'] = 0.03 * industries_ghgs.loc['BS33200']
            industries_ghgs.loc['BS332700'] = 0.16 * industries_ghgs.loc['BS33200']
            industries_ghgs.loc['BS332800'] = 0.06 * industries_ghgs.loc['BS33200']
            industries_ghgs.loc['BS333100'] = 0.25 * industries_ghgs.loc['BS33300']
            industries_ghgs.loc['BS333200'] = 0.12 * industries_ghgs.loc['BS33300']
            industries_ghgs.loc['BS333300'] = 0.14 * industries_ghgs.loc['BS33300']
            industries_ghgs.loc['BS333400'] = 0.10 * industries_ghgs.loc['BS33300']
            industries_ghgs.loc['BS333500'] = 0.12 * industries_ghgs.loc['BS33300']
            industries_ghgs.loc['BS333600'] = 0.04 * industries_ghgs.loc['BS33300']
            industries_ghgs.loc['BS333900'] = 0.22 * industries_ghgs.loc['BS33300']
            industries_ghgs.loc['BS334200'] = 0.25 * industries_ghgs.loc['BS334B0']
            industries_ghgs.loc['BS334A00'] = 0.46 * industries_ghgs.loc['BS334B0']
            industries_ghgs.loc['BS334400'] = 0.29 * industries_ghgs.loc['BS334B0']
            industries_ghgs.loc['BS335100'] = 0.12 * industries_ghgs.loc['BS335A0']
            industries_ghgs.loc['BS335300'] = 0.49 * industries_ghgs.loc['BS335A0']
            industries_ghgs.loc['BS335900'] = 0.39 * industries_ghgs.loc['BS335A0']
            industries_ghgs.loc['BS336110'] = 0.96 * industries_ghgs.loc['BS33610']
            industries_ghgs.loc['BS336120'] = 0.04 * industries_ghgs.loc['BS33610']
            industries_ghgs.loc['BS336310'] = 0.17 * industries_ghgs.loc['BS33630']
            industries_ghgs.loc['BS336320'] = 0.05 * industries_ghgs.loc['BS33630']
            industries_ghgs.loc['BS336330'] = 0.07 * industries_ghgs.loc['BS33630']
            industries_ghgs.loc['BS336340'] = 0.02 * industries_ghgs.loc['BS33630']
            industries_ghgs.loc['BS336350'] = 0.13 * industries_ghgs.loc['BS33630']
            industries_ghgs.loc['BS336360'] = 0.19 * industries_ghgs.loc['BS33630']
            industries_ghgs.loc['BS336370'] = 0.20 * industries_ghgs.loc['BS33630']
            industries_ghgs.loc['BS336390'] = 0.17 * industries_ghgs.loc['BS33630']
            industries_ghgs.loc['BS337100'] = 0.54 * industries_ghgs.loc['BS33700']
            industries_ghgs.loc['BS337200'] = 0.36 * industries_ghgs.loc['BS33700']
            industries_ghgs.loc['BS337900'] = 0.1 * industries_ghgs.loc['BS33700']
            industries_ghgs.loc['BS339100'] = 0.28 * industries_ghgs.loc['BS33900']
            industries_ghgs.loc['BS339900'] = 0.72 * industries_ghgs.loc['BS33900']
            industries_ghgs.loc['BS411000'] = 0.02 * industries_ghgs.loc['BS41000']
            industries_ghgs.loc['BS412000'] = 0.04 * industries_ghgs.loc['BS41000']
            industries_ghgs.loc['BS413000'] = 0.13 * industries_ghgs.loc['BS41000']
            industries_ghgs.loc['BS414000'] = 0.17 * industries_ghgs.loc['BS41000']
            industries_ghgs.loc['BS415000'] = 0.12 * industries_ghgs.loc['BS41000']
            industries_ghgs.loc['BS416000'] = 0.14 * industries_ghgs.loc['BS41000']
            industries_ghgs.loc['BS417000'] = 0.23 * industries_ghgs.loc['BS41000']
            industries_ghgs.loc['BS418000'] = 0.12 * industries_ghgs.loc['BS41000']
            industries_ghgs.loc['BS419000'] = 0.03 * industries_ghgs.loc['BS41000']
            industries_ghgs.loc['BS441000'] = 0.16 * industries_ghgs.loc['BS4AA00']
            industries_ghgs.loc['BS442000'] = 0.05 * industries_ghgs.loc['BS4AA00']
            industries_ghgs.loc['BS443000'] = 0.03 * industries_ghgs.loc['BS4AA00']
            industries_ghgs.loc['BS444000'] = 0.08 * industries_ghgs.loc['BS4AA00']
            industries_ghgs.loc['BS445000'] = 0.19 * industries_ghgs.loc['BS4AA00']
            industries_ghgs.loc['BS446000'] = 0.11 * industries_ghgs.loc['BS4AA00']
            industries_ghgs.loc['BS447000'] = 0.06 * industries_ghgs.loc['BS4AA00']
            industries_ghgs.loc['BS448000'] = 0.11 * industries_ghgs.loc['BS4AA00']
            industries_ghgs.loc['BS451000'] = 0.03 * industries_ghgs.loc['BS4AA00']
            industries_ghgs.loc['BS452000'] = 0.1 * industries_ghgs.loc['BS4AA00']
            industries_ghgs.loc['BS453A00'] = 0.04 * industries_ghgs.loc['BS4AA00']
            industries_ghgs.loc['BS454000'] = 0.04 * industries_ghgs.loc['BS4AA00']
            industries_ghgs.loc['BS486A00'] = 0.51 * industries_ghgs.loc['BS48600']
            industries_ghgs.loc['BS486200'] = 0.49 * industries_ghgs.loc['BS48600']
            industries_ghgs.loc['BS485100'] = 0.11 * industries_ghgs.loc['BS48B00']
            industries_ghgs.loc['BS48A000'] = 0.09 * industries_ghgs.loc['BS48B00']
            industries_ghgs.loc['BS485300'] = 0.06 * industries_ghgs.loc['BS48B00']
            industries_ghgs.loc['BS488000'] = 0.75 * industries_ghgs.loc['BS48B00']
            industries_ghgs.loc['BS491000'] = 0.34 * industries_ghgs.loc['BS49A00']
            industries_ghgs.loc['BS492000'] = 0.66 * industries_ghgs.loc['BS49A00']
            industries_ghgs.loc['BS5121A0'] = 0.77 * industries_ghgs.loc['BS51200']
            industries_ghgs.loc['BS512130'] = 0.14 * industries_ghgs.loc['BS51200']
            industries_ghgs.loc['BS512200'] = 0.08 * industries_ghgs.loc['BS51200']
            industries_ghgs.loc['BS511110'] = 0.04 * industries_ghgs.loc['BS51B00']
            industries_ghgs.loc['BS5111A0'] = 0.05 * industries_ghgs.loc['BS51B00']
            industries_ghgs.loc['BS511200'] = 0.11 * industries_ghgs.loc['BS51B00']
            industries_ghgs.loc['BS515200'] = 0.05 * industries_ghgs.loc['BS51B00']
            industries_ghgs.loc['BS517000'] = 0.66 * industries_ghgs.loc['BS51B00']
            industries_ghgs.loc['BS518000'] = 0.06 * industries_ghgs.loc['BS51B00']
            industries_ghgs.loc['BS519000'] = 0.04 * industries_ghgs.loc['BS51B00']
            industries_ghgs.loc['BS521000'] = 0.002 * industries_ghgs.loc['BS52B00']
            industries_ghgs.loc['BS5221A0'] = 0.469 * industries_ghgs.loc['BS52B00']
            industries_ghgs.loc['BS522130'] = 0.042 * industries_ghgs.loc['BS52B00']
            industries_ghgs.loc['BS522200'] = 0.067 * industries_ghgs.loc['BS52B00']
            industries_ghgs.loc['BS522300'] = 0.031 * industries_ghgs.loc['BS52B00']
            industries_ghgs.loc['BS52A000'] = 0.306 * industries_ghgs.loc['BS52B00']
            industries_ghgs.loc['BS524200'] = 0.082 * industries_ghgs.loc['BS52B00']
            industries_ghgs.loc['BS532100'] = 0.28 * industries_ghgs.loc['BS53B00']
            industries_ghgs.loc['BS532A00'] = 0.56 * industries_ghgs.loc['BS53B00']
            industries_ghgs.loc['BS533000'] = 0.16 * industries_ghgs.loc['BS53B00']
            industries_ghgs.loc['BS541100'] = 0.26 * industries_ghgs.loc['BS541C0']
            industries_ghgs.loc['BS541200'] = 0.26 * industries_ghgs.loc['BS541C0']
            industries_ghgs.loc['BS541300'] = 0.48 * industries_ghgs.loc['BS541C0']
            industries_ghgs.loc['BS541400'] = 0.03 * industries_ghgs.loc['BS541D0']
            industries_ghgs.loc['BS541500'] = 0.52 * industries_ghgs.loc['BS541D0']
            industries_ghgs.loc['BS541600'] = 0.21 * industries_ghgs.loc['BS541D0']
            industries_ghgs.loc['BS541700'] = 0.08 * industries_ghgs.loc['BS541D0']
            industries_ghgs.loc['BS541900'] = 0.16 * industries_ghgs.loc['BS541D0']
            industries_ghgs.loc['BS561100'] = 0.18 * industries_ghgs.loc['BS56100']
            industries_ghgs.loc['BS561A00'] = 0.16 * industries_ghgs.loc['BS56100']
            industries_ghgs.loc['BS561300'] = 0.17 * industries_ghgs.loc['BS56100']
            industries_ghgs.loc['BS561400'] = 0.11 * industries_ghgs.loc['BS56100']
            industries_ghgs.loc['BS561500'] = 0.07 * industries_ghgs.loc['BS56100']
            industries_ghgs.loc['BS561600'] = 0.09 * industries_ghgs.loc['BS56100']
            industries_ghgs.loc['BS561700'] = 0.23 * industries_ghgs.loc['BS56100']
            industries_ghgs.loc['BS531A00'] = 0.62 * industries_ghgs.loc['BS5A000']
            industries_ghgs.loc['BS551113'] = 0.38 * industries_ghgs.loc['BS5A000']
            industries_ghgs.loc['BS621100'] = 0.44 * industries_ghgs.loc['BS62000']
            industries_ghgs.loc['BS621200'] = 0.21 * industries_ghgs.loc['BS62000']
            industries_ghgs.loc['BS621A00'] = 0.17 * industries_ghgs.loc['BS62000']
            industries_ghgs.loc['BS623000'] = 0.11 * industries_ghgs.loc['BS62000']
            industries_ghgs.loc['BS624000'] = 0.07 * industries_ghgs.loc['BS62000']
            industries_ghgs.loc['BS71A000'] = 0.37 * industries_ghgs.loc['BS71000']
            industries_ghgs.loc['BS713A00'] = 0.37 * industries_ghgs.loc['BS71000']
            industries_ghgs.loc['BS713200'] = 0.26 * industries_ghgs.loc['BS71000']
            industries_ghgs.loc['BS721100'] = 0.19 * industries_ghgs.loc['BS72000']
            industries_ghgs.loc['BS721A00'] = 0.03 * industries_ghgs.loc['BS72000']
            industries_ghgs.loc['BS722000'] = 0.78 * industries_ghgs.loc['BS72000']
            industries_ghgs.loc['BS811100'] = 0.53 * industries_ghgs.loc['BS81100']
            industries_ghgs.loc['BS811A00'] = 0.47 * industries_ghgs.loc['BS81100']
            industries_ghgs.loc['BS812A00'] = 0.59 * industries_ghgs.loc['BS81A00']
            industries_ghgs.loc['BS812200'] = 0.11 * industries_ghgs.loc['BS81A00']
            industries_ghgs.loc['BS812300'] = 0.12 * industries_ghgs.loc['BS81A00']
            industries_ghgs.loc['BS814000'] = 0.18 * industries_ghgs.loc['BS81A00']
            industries_ghgs.loc['FC210000'] = 0.5 * industries_ghgs.loc['FC20000']
            industries_ghgs.loc['FC220000'] = 0.5 * industries_ghgs.loc['FC20000']
            industries_ghgs.loc['NP621000'] = 0.09 * industries_ghgs.loc['NPA0000']
            industries_ghgs.loc['NP813A00'] = 0.65 * industries_ghgs.loc['NPA0000']
            industries_ghgs.loc['NP999999'] = 0.26 * industries_ghgs.loc['NPA0000']
            industries_ghgs.loc['GS611100'] = 0.83 * industries_ghgs.loc['GS611B0']
            industries_ghgs.loc['GS611200'] = 0.16 * industries_ghgs.loc['GS611B0']
            industries_ghgs.loc['GS611A00'] = 0.01 * industries_ghgs.loc['GS611B0']
            industries_ghgs.loc['GS911100'] = 0.27 * industries_ghgs.loc['GS91100']
            industries_ghgs.loc['GS911A00'] = 0.73 * industries_ghgs.loc['GS91100']

            new_index = []
            for code in industries_ghgs.index:
                if code in key_changes:
                    if key_changes[code] != '':
                        new_index.append(key_changes[code])
                    else:
                        new_index.append(code)
                else:
                    new_index.append(code)

            industries_ghgs.index = new_index
            industries_ghgs = industries_ghgs.loc[IOIC_codes]

        industries_ghgs.index = [i[1] for i in self.industries]
        industries_ghgs.name = ('GHGs', '')
        industries_ghgs *= 1000000
        self.F = self.F.append(industries_ghgs)

        self.emission_metadata.loc[('GHGs', ''), 'CAS Number'] = 'N/A'
        self.emission_metadata.loc[('GHGs', ''), 'Unit'] = 'kgCO2eq'

        # ---------------------Energy use-------------------------

        if self.level_of_detail == 'Link-1961 level':
            industries_nrg = industries_nrg.drop([i for i in industries_nrg.index if i not in IOIC_codes])
        elif self.level_of_detail == 'Link-1997 level':
            key_changes = dict.fromkeys(industries_nrg.index, '')
            for key in key_changes:
                if key + '0' in IOIC_codes:
                    key_changes[key] = key + '0'

            # economic allocations. Be my guest if you wanna verify :)
            industries_nrg.loc['BS111A00'] = 0.55 * industries_nrg.loc['BS11B00']
            industries_nrg.loc['BS1114A0'] = 0.05 * industries_nrg.loc['BS11B00']
            industries_nrg.loc['BS112000'] = 0.40 * industries_nrg.loc['BS11B00']
            industries_nrg.loc['BS115A00'] = 0.39 * industries_nrg.loc['BS11500']
            industries_nrg.loc['BS115300'] = 0.61 * industries_nrg.loc['BS11500']
            industries_nrg.loc['BS212210'] = 0.15 * industries_nrg.loc['BS21220']
            industries_nrg.loc['BS212220'] = 0.35 * industries_nrg.loc['BS21220']
            industries_nrg.loc['BS212230'] = 0.4 * industries_nrg.loc['BS21220']
            industries_nrg.loc['BS212290'] = 0.1 * industries_nrg.loc['BS21220']
            industries_nrg.loc['BS212310'] = 0.16 * industries_nrg.loc['BS21230']
            industries_nrg.loc['BS212320'] = 0.15 * industries_nrg.loc['BS21230']
            industries_nrg.loc['BS212392'] = 0.18 * industries_nrg.loc['BS21230']
            industries_nrg.loc['BS21239A'] = 0.13 * industries_nrg.loc['BS21230']
            industries_nrg.loc['BS212396'] = 0.38 * industries_nrg.loc['BS21230']
            industries_nrg.loc['BS221200'] = 0.9 * industries_nrg.loc['BS221A0']
            industries_nrg.loc['BS221300'] = 0.1 * industries_nrg.loc['BS221A0']
            industries_nrg.loc['BS311200'] = 0.35 * industries_nrg.loc['BS311A0']
            industries_nrg.loc['BS311800'] = 0.31 * industries_nrg.loc['BS311A0']
            industries_nrg.loc['BS311900'] = 0.33 * industries_nrg.loc['BS311A0']
            industries_nrg.loc['BS321100'] = 0.52 * industries_nrg.loc['BS32100']
            industries_nrg.loc['BS321200'] = 0.22 * industries_nrg.loc['BS32100']
            industries_nrg.loc['BS321900'] = 0.26 * industries_nrg.loc['BS32100']
            industries_nrg.loc['BS325B00'] = 0.57 * industries_nrg.loc['BS325C0']
            industries_nrg.loc['BS325600'] = 0.20 * industries_nrg.loc['BS325C0']
            industries_nrg.loc['BS325900'] = 0.23 * industries_nrg.loc['BS325C0']
            industries_nrg.loc['BS331100'] = 0.17 * industries_nrg.loc['BS33100']
            industries_nrg.loc['BS331200'] = 0.07 * industries_nrg.loc['BS33100']
            industries_nrg.loc['BS331300'] = 0.18 * industries_nrg.loc['BS33100']
            industries_nrg.loc['BS331400'] = 0.54 * industries_nrg.loc['BS33100']
            industries_nrg.loc['BS331500'] = 0.04 * industries_nrg.loc['BS33100']
            industries_nrg.loc['BS332100'] = 0.04 * industries_nrg.loc['BS33200']
            industries_nrg.loc['BS332A00'] = 0.16 * industries_nrg.loc['BS33200']
            industries_nrg.loc['BS332300'] = 0.41 * industries_nrg.loc['BS33200']
            industries_nrg.loc['BS332400'] = 0.1 * industries_nrg.loc['BS33200']
            industries_nrg.loc['BS332500'] = 0.05 * industries_nrg.loc['BS33200']
            industries_nrg.loc['BS332600'] = 0.03 * industries_nrg.loc['BS33200']
            industries_nrg.loc['BS332700'] = 0.16 * industries_nrg.loc['BS33200']
            industries_nrg.loc['BS332800'] = 0.06 * industries_nrg.loc['BS33200']
            industries_nrg.loc['BS333100'] = 0.25 * industries_nrg.loc['BS33300']
            industries_nrg.loc['BS333A00'] = 0.26 * industries_nrg.loc['BS33300']
            industries_nrg.loc['BS333400'] = 0.1 * industries_nrg.loc['BS33300']
            industries_nrg.loc['BS333500'] = 0.12 * industries_nrg.loc['BS33300']
            industries_nrg.loc['BS333600'] = 0.04 * industries_nrg.loc['BS33300']
            industries_nrg.loc['BS333900'] = 0.22 * industries_nrg.loc['BS33300']
            industries_nrg.loc['BS334200'] = 0.25 * industries_nrg.loc['BS334B0']
            industries_nrg.loc['BS334A00'] = 0.46 * industries_nrg.loc['BS334B0']
            industries_nrg.loc['BS334400'] = 0.29 * industries_nrg.loc['BS334B0']
            industries_nrg.loc['BS335100'] = 0.12 * industries_nrg.loc['BS335A0']
            industries_nrg.loc['BS335300'] = 0.49 * industries_nrg.loc['BS335A0']
            industries_nrg.loc['BS335900'] = 0.39 * industries_nrg.loc['BS335A0']
            industries_nrg.loc['BS337100'] = 0.54 * industries_nrg.loc['BS33700']
            industries_nrg.loc['BS337200'] = 0.36 * industries_nrg.loc['BS33700']
            industries_nrg.loc['BS337900'] = 0.1 * industries_nrg.loc['BS33700']
            industries_nrg.loc['BS339100'] = 0.28 * industries_nrg.loc['BS33900']
            industries_nrg.loc['BS339900'] = 0.72 * industries_nrg.loc['BS33900']
            industries_nrg.loc['BS486A00'] = 0.51 * industries_nrg.loc['BS48600']
            industries_nrg.loc['BS486200'] = 0.49 * industries_nrg.loc['BS48600']
            industries_nrg.loc['BS485100'] = 0.11 * industries_nrg.loc['BS48B00']
            industries_nrg.loc['BS48A000'] = 0.09 * industries_nrg.loc['BS48B00']
            industries_nrg.loc['BS485300'] = 0.06 * industries_nrg.loc['BS48B00']
            industries_nrg.loc['BS488000'] = 0.75 * industries_nrg.loc['BS48B00']
            industries_nrg.loc['BS5121A0'] = 0.77 * industries_nrg.loc['BS51200']
            industries_nrg.loc['BS512130'] = 0.14 * industries_nrg.loc['BS51200']
            industries_nrg.loc['BS512200'] = 0.08 * industries_nrg.loc['BS51200']
            industries_nrg.loc['BS511100'] = 0.09 * industries_nrg.loc['BS51B00']
            industries_nrg.loc['BS511200'] = 0.11 * industries_nrg.loc['BS51B00']
            industries_nrg.loc['BS51A000'] = 0.74 * industries_nrg.loc['BS51B00']
            industries_nrg.loc['BS518000'] = 0.06 * industries_nrg.loc['BS51B00']
            industries_nrg.loc['BS521000'] = 0.002 * industries_nrg.loc['BS52B00']
            industries_nrg.loc['BS5221A0'] = 0.469 * industries_nrg.loc['BS52B00']
            industries_nrg.loc['BS522130'] = 0.042 * industries_nrg.loc['BS52B00']
            industries_nrg.loc['BS522A00'] = 0.099 * industries_nrg.loc['BS52B00']
            industries_nrg.loc['BS52A000'] = 0.306 * industries_nrg.loc['BS52B00']
            industries_nrg.loc['BS524200'] = 0.082 * industries_nrg.loc['BS52B00']
            industries_nrg.loc['BS532100'] = 0.28 * industries_nrg.loc['BS53B00']
            industries_nrg.loc['BS53A000'] = 0.72 * industries_nrg.loc['BS53B00']
            industries_nrg.loc['BS541A00'] = 0.52 * industries_nrg.loc['BS541C0']
            industries_nrg.loc['BS541300'] = 0.48 * industries_nrg.loc['BS541C0']
            industries_nrg.loc['BS541B00'] = 0.48 * industries_nrg.loc['BS541D0']
            industries_nrg.loc['BS541500'] = 0.52 * industries_nrg.loc['BS541D0']
            industries_nrg.loc['BS561B00'] = 0.61 * industries_nrg.loc['BS56100']
            industries_nrg.loc['BS561500'] = 0.06 * industries_nrg.loc['BS56100']
            industries_nrg.loc['BS561600'] = 0.09 * industries_nrg.loc['BS56100']
            industries_nrg.loc['BS561700'] = 0.23 * industries_nrg.loc['BS56100']
            industries_nrg.loc['BS531A00'] = 0.62 * industries_nrg.loc['BS5A000']
            industries_nrg.loc['BS551113'] = 0.38 * industries_nrg.loc['BS5A000']
            industries_nrg.loc['BS621100'] = 0.44 * industries_nrg.loc['BS62000']
            industries_nrg.loc['BS621200'] = 0.21 * industries_nrg.loc['BS62000']
            industries_nrg.loc['BS621A00'] = 0.17 * industries_nrg.loc['BS62000']
            industries_nrg.loc['BS623000'] = 0.11 * industries_nrg.loc['BS62000']
            industries_nrg.loc['BS624000'] = 0.07 * industries_nrg.loc['BS62000']
            industries_nrg.loc['BS71A000'] = 0.37 * industries_nrg.loc['BS71000']
            industries_nrg.loc['BS713A00'] = 0.37 * industries_nrg.loc['BS71000']
            industries_nrg.loc['BS713200'] = 0.26 * industries_nrg.loc['BS71000']
            industries_nrg.loc['BS721100'] = 0.19 * industries_nrg.loc['BS72000']
            industries_nrg.loc['BS721A00'] = 0.03 * industries_nrg.loc['BS72000']
            industries_nrg.loc['BS722000'] = 0.78 * industries_nrg.loc['BS72000']
            industries_nrg.loc['BS811100'] = 0.53 * industries_nrg.loc['BS81100']
            industries_nrg.loc['BS811A00'] = 0.47 * industries_nrg.loc['BS81100']
            industries_nrg.loc['BS812A00'] = 0.59 * industries_nrg.loc['BS81A00']
            industries_nrg.loc['BS812200'] = 0.11 * industries_nrg.loc['BS81A00']
            industries_nrg.loc['BS812300'] = 0.12 * industries_nrg.loc['BS81A00']
            industries_nrg.loc['BS814000'] = 0.18 * industries_nrg.loc['BS81A00']
            industries_nrg.loc['GS611100'] = 0.83 * industries_nrg.loc['GS611B0']
            industries_nrg.loc['GS611200'] = 0.16 * industries_nrg.loc['GS611B0']
            industries_nrg.loc['GS611A00'] = 0.01 * industries_nrg.loc['GS611B0']
            industries_nrg.loc['GS911100'] = 0.27 * industries_nrg.loc['GS91100']
            industries_nrg.loc['GS911A00'] = 0.73 * industries_nrg.loc['GS91100']

            new_index = []
            for code in industries_nrg.index:
                if code in key_changes:
                    if key_changes[code] != '':
                        new_index.append(key_changes[code])
                    else:
                        new_index.append(code)
                else:
                    new_index.append(code)

            industries_nrg.index = new_index
            industries_nrg = industries_nrg.loc[IOIC_codes]
        elif self.level_of_detail == 'Summary level':
            industries_nrg.index = [i[:-2] for i in industries_nrg.index]
            industries_nrg = industries_nrg.groupby(industries_nrg.index).sum()

            industries_nrg.loc['BS11A'] += industries_nrg.loc[['BS11B', 'BS111']].sum()
            industries_nrg.loc['BS210'] = industries_nrg.loc[['BS211', 'BS212', 'BS213']].sum()
            industries_nrg.loc['BS220'] = industries_nrg.loc[['BS221']].sum()
            industries_nrg.loc['BS3A0'] = industries_nrg.loc[
                ['BS311', 'BS312', 'BS31A', 'BS31B', 'BS321', 'BS322', 'BS323',
                 'BS324', 'BS325', 'BS326', 'BS327', 'BS331', 'BS332', 'BS333',
                 'BS334', 'BS335', 'BS336', 'BS337', 'BS339']].sum()
            industries_nrg.loc['BS4A0'] = industries_nrg.loc[['BS4AA', 'BS453']].sum()
            industries_nrg.loc['BS4B0'] = industries_nrg.loc[
                ['BS481', 'BS482', 'BS483', 'BS484', 'BS48B', 'BS486', 'BS49A', 'BS493']].sum()
            industries_nrg.loc['BS510'] = industries_nrg.loc[['BS512', 'BS515', 'BS51B']].sum()
            industries_nrg.loc['BS5B0'] = industries_nrg.loc[['BS52B', 'BS524', 'BS531', 'BS53B', 'BS5A0']].sum()
            industries_nrg.loc['BS540'] = industries_nrg.loc[['BS541']].sum()
            industries_nrg.loc['BS560'] = industries_nrg.loc[['BS561', 'BS562']].sum()
            industries_nrg.loc['BS810'] = industries_nrg.loc[['BS811', 'BS81A', 'BS813']].sum()
            industries_nrg.loc['FC100'] = industries_nrg.loc[['FC110', 'FC120', 'FC130']].sum()
            industries_nrg.loc['NP000'] = industries_nrg.loc[['NP610', 'NP624', 'NP710', 'NP813', 'NPA00']].sum()
            industries_nrg.loc['GS610'] = industries_nrg.loc[['GS611']].sum()
            industries_nrg.loc['GS620'] = industries_nrg.loc[['GS622', 'GS623']].sum()

            industries_nrg = industries_nrg.reindex(IOIC_codes).fillna(0)
        elif self.level_of_detail == 'Detail level':
            key_changes = dict.fromkeys(industries_nrg.index, '')
            for key in key_changes:
                if key + '0' in IOIC_codes:
                    key_changes[key] = key + '0'

            # economic allocations. Be my guest if you wanna verify :)
            industries_nrg.loc['BS111A00'] = 0.55 * industries_nrg.loc['BS11B00']
            industries_nrg.loc['BS1114A0'] = 0.05 * industries_nrg.loc['BS11B00']
            industries_nrg.loc['BS112A00'] = 0.38 * industries_nrg.loc['BS11B00']
            industries_nrg.loc['BS112500'] = 0.02 * industries_nrg.loc['BS11B00']
            industries_nrg.loc['BS115A00'] = 0.39 * industries_nrg.loc['BS11500']
            industries_nrg.loc['BS115300'] = 0.61 * industries_nrg.loc['BS11500']
            industries_nrg.loc['BS211110'] = 0.5 * industries_nrg.loc['BS21100']
            industries_nrg.loc['BS211140'] = 0.5 * industries_nrg.loc['BS21100']
            industries_nrg.loc['BS212210'] = 0.15 * industries_nrg.loc['BS21220']
            industries_nrg.loc['BS212220'] = 0.35 * industries_nrg.loc['BS21220']
            industries_nrg.loc['BS212230'] = 0.4 * industries_nrg.loc['BS21220']
            industries_nrg.loc['BS212290'] = 0.1 * industries_nrg.loc['BS21220']
            industries_nrg.loc['BS212310'] = 0.16 * industries_nrg.loc['BS21230']
            industries_nrg.loc['BS212320'] = 0.15 * industries_nrg.loc['BS21230']
            industries_nrg.loc['BS212392'] = 0.18 * industries_nrg.loc['BS21230']
            industries_nrg.loc['BS21239A'] = 0.13 * industries_nrg.loc['BS21230']
            industries_nrg.loc['BS212396'] = 0.38 * industries_nrg.loc['BS21230']
            industries_nrg.loc['BS21311A'] = 0.73 * industries_nrg.loc['BS21300']
            industries_nrg.loc['BS21311B'] = 0.27 * industries_nrg.loc['BS21300']
            industries_nrg.loc['BS221200'] = 0.9 * industries_nrg.loc['BS221A0']
            industries_nrg.loc['BS221300'] = 0.1 * industries_nrg.loc['BS221A0']
            industries_nrg.loc['BS311200'] = 0.35 * industries_nrg.loc['BS311A0']
            industries_nrg.loc['BS311800'] = 0.31 * industries_nrg.loc['BS311A0']
            industries_nrg.loc['BS311900'] = 0.33 * industries_nrg.loc['BS311A0']
            industries_nrg.loc['BS321100'] = 0.52 * industries_nrg.loc['BS32100']
            industries_nrg.loc['BS321200'] = 0.22 * industries_nrg.loc['BS32100']
            industries_nrg.loc['BS321900'] = 0.26 * industries_nrg.loc['BS32100']
            industries_nrg.loc['BS324110'] = 0.91 * industries_nrg.loc['BS32400']
            industries_nrg.loc['BS3241A0'] = 0.09 * industries_nrg.loc['BS32400']
            industries_nrg.loc['BS325200'] = 0.43 * industries_nrg.loc['BS325C0']
            industries_nrg.loc['BS325500'] = 0.13 * industries_nrg.loc['BS325C0']
            industries_nrg.loc['BS325600'] = 0.20 * industries_nrg.loc['BS325C0']
            industries_nrg.loc['BS325900'] = 0.23 * industries_nrg.loc['BS325C0']
            industries_nrg.loc['BS331100'] = 0.17 * industries_nrg.loc['BS33100']
            industries_nrg.loc['BS331200'] = 0.07 * industries_nrg.loc['BS33100']
            industries_nrg.loc['BS331300'] = 0.18 * industries_nrg.loc['BS33100']
            industries_nrg.loc['BS331400'] = 0.54 * industries_nrg.loc['BS33100']
            industries_nrg.loc['BS331500'] = 0.04 * industries_nrg.loc['BS33100']
            industries_nrg.loc['BS332100'] = 0.04 * industries_nrg.loc['BS33200']
            industries_nrg.loc['BS332A00'] = 0.16 * industries_nrg.loc['BS33200']
            industries_nrg.loc['BS332300'] = 0.41 * industries_nrg.loc['BS33200']
            industries_nrg.loc['BS332400'] = 0.1 * industries_nrg.loc['BS33200']
            industries_nrg.loc['BS332500'] = 0.05 * industries_nrg.loc['BS33200']
            industries_nrg.loc['BS332600'] = 0.03 * industries_nrg.loc['BS33200']
            industries_nrg.loc['BS332700'] = 0.16 * industries_nrg.loc['BS33200']
            industries_nrg.loc['BS332800'] = 0.06 * industries_nrg.loc['BS33200']
            industries_nrg.loc['BS333100'] = 0.25 * industries_nrg.loc['BS33300']
            industries_nrg.loc['BS333200'] = 0.12 * industries_nrg.loc['BS33300']
            industries_nrg.loc['BS333300'] = 0.14 * industries_nrg.loc['BS33300']
            industries_nrg.loc['BS333400'] = 0.10 * industries_nrg.loc['BS33300']
            industries_nrg.loc['BS333500'] = 0.12 * industries_nrg.loc['BS33300']
            industries_nrg.loc['BS333600'] = 0.04 * industries_nrg.loc['BS33300']
            industries_nrg.loc['BS333900'] = 0.22 * industries_nrg.loc['BS33300']
            industries_nrg.loc['BS334200'] = 0.25 * industries_nrg.loc['BS334B0']
            industries_nrg.loc['BS334A00'] = 0.46 * industries_nrg.loc['BS334B0']
            industries_nrg.loc['BS334400'] = 0.29 * industries_nrg.loc['BS334B0']
            industries_nrg.loc['BS335100'] = 0.12 * industries_nrg.loc['BS335A0']
            industries_nrg.loc['BS335300'] = 0.49 * industries_nrg.loc['BS335A0']
            industries_nrg.loc['BS335900'] = 0.39 * industries_nrg.loc['BS335A0']
            industries_nrg.loc['BS336110'] = 0.96 * industries_nrg.loc['BS33610']
            industries_nrg.loc['BS336120'] = 0.04 * industries_nrg.loc['BS33610']
            industries_nrg.loc['BS336310'] = 0.17 * industries_nrg.loc['BS33630']
            industries_nrg.loc['BS336320'] = 0.05 * industries_nrg.loc['BS33630']
            industries_nrg.loc['BS336330'] = 0.07 * industries_nrg.loc['BS33630']
            industries_nrg.loc['BS336340'] = 0.02 * industries_nrg.loc['BS33630']
            industries_nrg.loc['BS336350'] = 0.13 * industries_nrg.loc['BS33630']
            industries_nrg.loc['BS336360'] = 0.19 * industries_nrg.loc['BS33630']
            industries_nrg.loc['BS336370'] = 0.20 * industries_nrg.loc['BS33630']
            industries_nrg.loc['BS336390'] = 0.17 * industries_nrg.loc['BS33630']
            industries_nrg.loc['BS337100'] = 0.54 * industries_nrg.loc['BS33700']
            industries_nrg.loc['BS337200'] = 0.36 * industries_nrg.loc['BS33700']
            industries_nrg.loc['BS337900'] = 0.1 * industries_nrg.loc['BS33700']
            industries_nrg.loc['BS339100'] = 0.28 * industries_nrg.loc['BS33900']
            industries_nrg.loc['BS339900'] = 0.72 * industries_nrg.loc['BS33900']
            industries_nrg.loc['BS411000'] = 0.02 * industries_nrg.loc['BS41000']
            industries_nrg.loc['BS412000'] = 0.04 * industries_nrg.loc['BS41000']
            industries_nrg.loc['BS413000'] = 0.13 * industries_nrg.loc['BS41000']
            industries_nrg.loc['BS414000'] = 0.17 * industries_nrg.loc['BS41000']
            industries_nrg.loc['BS415000'] = 0.12 * industries_nrg.loc['BS41000']
            industries_nrg.loc['BS416000'] = 0.14 * industries_nrg.loc['BS41000']
            industries_nrg.loc['BS417000'] = 0.23 * industries_nrg.loc['BS41000']
            industries_nrg.loc['BS418000'] = 0.12 * industries_nrg.loc['BS41000']
            industries_nrg.loc['BS419000'] = 0.03 * industries_nrg.loc['BS41000']
            industries_nrg.loc['BS441000'] = 0.16 * industries_nrg.loc['BS4AA00']
            industries_nrg.loc['BS442000'] = 0.05 * industries_nrg.loc['BS4AA00']
            industries_nrg.loc['BS443000'] = 0.03 * industries_nrg.loc['BS4AA00']
            industries_nrg.loc['BS444000'] = 0.08 * industries_nrg.loc['BS4AA00']
            industries_nrg.loc['BS445000'] = 0.19 * industries_nrg.loc['BS4AA00']
            industries_nrg.loc['BS446000'] = 0.11 * industries_nrg.loc['BS4AA00']
            industries_nrg.loc['BS447000'] = 0.06 * industries_nrg.loc['BS4AA00']
            industries_nrg.loc['BS448000'] = 0.11 * industries_nrg.loc['BS4AA00']
            industries_nrg.loc['BS451000'] = 0.03 * industries_nrg.loc['BS4AA00']
            industries_nrg.loc['BS452000'] = 0.1 * industries_nrg.loc['BS4AA00']
            industries_nrg.loc['BS453A00'] = 0.04 * industries_nrg.loc['BS4AA00']
            industries_nrg.loc['BS454000'] = 0.04 * industries_nrg.loc['BS4AA00']
            industries_nrg.loc['BS486A00'] = 0.51 * industries_nrg.loc['BS48600']
            industries_nrg.loc['BS486200'] = 0.49 * industries_nrg.loc['BS48600']
            industries_nrg.loc['BS485100'] = 0.11 * industries_nrg.loc['BS48B00']
            industries_nrg.loc['BS48A000'] = 0.09 * industries_nrg.loc['BS48B00']
            industries_nrg.loc['BS485300'] = 0.06 * industries_nrg.loc['BS48B00']
            industries_nrg.loc['BS488000'] = 0.75 * industries_nrg.loc['BS48B00']
            industries_nrg.loc['BS491000'] = 0.34 * industries_nrg.loc['BS49A00']
            industries_nrg.loc['BS492000'] = 0.66 * industries_nrg.loc['BS49A00']
            industries_nrg.loc['BS5121A0'] = 0.77 * industries_nrg.loc['BS51200']
            industries_nrg.loc['BS512130'] = 0.14 * industries_nrg.loc['BS51200']
            industries_nrg.loc['BS512200'] = 0.08 * industries_nrg.loc['BS51200']
            industries_nrg.loc['BS511110'] = 0.04 * industries_nrg.loc['BS51B00']
            industries_nrg.loc['BS5111A0'] = 0.05 * industries_nrg.loc['BS51B00']
            industries_nrg.loc['BS511200'] = 0.11 * industries_nrg.loc['BS51B00']
            industries_nrg.loc['BS515200'] = 0.05 * industries_nrg.loc['BS51B00']
            industries_nrg.loc['BS517000'] = 0.66 * industries_nrg.loc['BS51B00']
            industries_nrg.loc['BS518000'] = 0.06 * industries_nrg.loc['BS51B00']
            industries_nrg.loc['BS519000'] = 0.04 * industries_nrg.loc['BS51B00']
            industries_nrg.loc['BS521000'] = 0.002 * industries_nrg.loc['BS52B00']
            industries_nrg.loc['BS5221A0'] = 0.469 * industries_nrg.loc['BS52B00']
            industries_nrg.loc['BS522130'] = 0.042 * industries_nrg.loc['BS52B00']
            industries_nrg.loc['BS522200'] = 0.067 * industries_nrg.loc['BS52B00']
            industries_nrg.loc['BS522300'] = 0.031 * industries_nrg.loc['BS52B00']
            industries_nrg.loc['BS52A000'] = 0.306 * industries_nrg.loc['BS52B00']
            industries_nrg.loc['BS524200'] = 0.082 * industries_nrg.loc['BS52B00']
            industries_nrg.loc['BS532100'] = 0.28 * industries_nrg.loc['BS53B00']
            industries_nrg.loc['BS532A00'] = 0.56 * industries_nrg.loc['BS53B00']
            industries_nrg.loc['BS533000'] = 0.16 * industries_nrg.loc['BS53B00']
            industries_nrg.loc['BS541100'] = 0.26 * industries_nrg.loc['BS541C0']
            industries_nrg.loc['BS541200'] = 0.26 * industries_nrg.loc['BS541C0']
            industries_nrg.loc['BS541300'] = 0.48 * industries_nrg.loc['BS541C0']
            industries_nrg.loc['BS541400'] = 0.03 * industries_nrg.loc['BS541D0']
            industries_nrg.loc['BS541500'] = 0.52 * industries_nrg.loc['BS541D0']
            industries_nrg.loc['BS541600'] = 0.21 * industries_nrg.loc['BS541D0']
            industries_nrg.loc['BS541700'] = 0.08 * industries_nrg.loc['BS541D0']
            industries_nrg.loc['BS541900'] = 0.16 * industries_nrg.loc['BS541D0']
            industries_nrg.loc['BS561100'] = 0.18 * industries_nrg.loc['BS56100']
            industries_nrg.loc['BS561A00'] = 0.16 * industries_nrg.loc['BS56100']
            industries_nrg.loc['BS561300'] = 0.17 * industries_nrg.loc['BS56100']
            industries_nrg.loc['BS561400'] = 0.11 * industries_nrg.loc['BS56100']
            industries_nrg.loc['BS561500'] = 0.07 * industries_nrg.loc['BS56100']
            industries_nrg.loc['BS561600'] = 0.09 * industries_nrg.loc['BS56100']
            industries_nrg.loc['BS561700'] = 0.23 * industries_nrg.loc['BS56100']
            industries_nrg.loc['BS531A00'] = 0.62 * industries_nrg.loc['BS5A000']
            industries_nrg.loc['BS551113'] = 0.38 * industries_nrg.loc['BS5A000']
            industries_nrg.loc['BS621100'] = 0.44 * industries_nrg.loc['BS62000']
            industries_nrg.loc['BS621200'] = 0.21 * industries_nrg.loc['BS62000']
            industries_nrg.loc['BS621A00'] = 0.17 * industries_nrg.loc['BS62000']
            industries_nrg.loc['BS623000'] = 0.11 * industries_nrg.loc['BS62000']
            industries_nrg.loc['BS624000'] = 0.07 * industries_nrg.loc['BS62000']
            industries_nrg.loc['BS71A000'] = 0.37 * industries_nrg.loc['BS71000']
            industries_nrg.loc['BS713A00'] = 0.37 * industries_nrg.loc['BS71000']
            industries_nrg.loc['BS713200'] = 0.26 * industries_nrg.loc['BS71000']
            industries_nrg.loc['BS721100'] = 0.19 * industries_nrg.loc['BS72000']
            industries_nrg.loc['BS721A00'] = 0.03 * industries_nrg.loc['BS72000']
            industries_nrg.loc['BS722000'] = 0.78 * industries_nrg.loc['BS72000']
            industries_nrg.loc['BS811100'] = 0.53 * industries_nrg.loc['BS81100']
            industries_nrg.loc['BS811A00'] = 0.47 * industries_nrg.loc['BS81100']
            industries_nrg.loc['BS812A00'] = 0.59 * industries_nrg.loc['BS81A00']
            industries_nrg.loc['BS812200'] = 0.11 * industries_nrg.loc['BS81A00']
            industries_nrg.loc['BS812300'] = 0.12 * industries_nrg.loc['BS81A00']
            industries_nrg.loc['BS814000'] = 0.18 * industries_nrg.loc['BS81A00']
            industries_nrg.loc['FC210000'] = 0.5 * industries_nrg.loc['FC20000']
            industries_nrg.loc['FC220000'] = 0.5 * industries_nrg.loc['FC20000']
            industries_nrg.loc['NP621000'] = 0.09 * industries_nrg.loc['NPA0000']
            industries_nrg.loc['NP813A00'] = 0.65 * industries_nrg.loc['NPA0000']
            industries_nrg.loc['NP999999'] = 0.26 * industries_nrg.loc['NPA0000']
            industries_nrg.loc['GS611100'] = 0.83 * industries_nrg.loc['GS611B0']
            industries_nrg.loc['GS611200'] = 0.16 * industries_nrg.loc['GS611B0']
            industries_nrg.loc['GS611A00'] = 0.01 * industries_nrg.loc['GS611B0']
            industries_nrg.loc['GS911100'] = 0.27 * industries_nrg.loc['GS91100']
            industries_nrg.loc['GS911A00'] = 0.73 * industries_nrg.loc['GS91100']

            new_index = []
            for code in industries_nrg.index:
                if code in key_changes:
                    if key_changes[code] != '':
                        new_index.append(key_changes[code])
                    else:
                        new_index.append(code)
                else:
                    new_index.append(code)

            industries_nrg.index = new_index
            industries_nrg = industries_nrg.loc[IOIC_codes]

        industries_nrg.index = [i[1] for i in self.industries]
        industries_nrg.name = ('Energy use', '')
        self.F = self.F.append(industries_nrg)

        self.emission_metadata.loc[('Energy use', ''), 'CAS Number'] = 'N/A'
        self.emission_metadata.loc[('Energy use', ''), 'Unit'] = 'TJ'

        # ---------------------Water use-------------------------

        if self.level_of_detail == 'Link-1961 level':
            industries_water.loc['BS11B00'] = 0.93 * (industries_water.loc['BS111'] + industries_water.loc['BS112'])
            industries_water.loc['BS111CL'] = 0.01 * (industries_water.loc['BS111'] + industries_water.loc['BS112'])
            industries_water.loc['BS111CU'] = 0.06 * (industries_water.loc['BS111'] + industries_water.loc['BS112'])
            industries_water.loc['BS31110'] = 0.08 * industries_water.loc['BS311']
            industries_water.loc['BS31130'] = 0.04 * industries_water.loc['BS311']
            industries_water.loc['BS31140'] = 0.07 * industries_water.loc['BS311']
            industries_water.loc['BS31150'] = 0.15 * industries_water.loc['BS311']
            industries_water.loc['BS31160'] = 0.28 * industries_water.loc['BS311']
            industries_water.loc['BS31170'] = 0.06 * industries_water.loc['BS311']
            industries_water.loc['BS311A0'] = 0.32 * industries_water.loc['BS311']
            industries_water.loc['BS31211'] = 0.30 * industries_water.loc['BS312']
            industries_water.loc['BS31212'] = 0.40 * industries_water.loc['BS312']
            industries_water.loc['BS3121A'] = 0.16 * industries_water.loc['BS312']
            industries_water.loc['BS31220'] = 0.14 * industries_water.loc['BS312']
            industries_water.loc['BS31A00'] = industries_water.loc['BS31A']
            industries_water.loc['BS31B00'] = industries_water.loc['BS31B']
            industries_water.loc['BS32100'] = industries_water.loc['BS321']
            industries_water.loc['BS32210'] = 0.62 * industries_water.loc['BS322']
            industries_water.loc['BS32220'] = 0.38 * industries_water.loc['BS322']
            industries_water.loc['BS32300'] = industries_water.loc['BS323']
            industries_water.loc['BS32400'] = industries_water.loc['BS324']
            industries_water.loc['BS32510'] = 0.27 * industries_water.loc['BS325']
            industries_water.loc['BS32530'] = 0.10 * industries_water.loc['BS325']
            industries_water.loc['BS32540'] = 0.23 * industries_water.loc['BS325']
            industries_water.loc['BS325C0'] = 0.40 * industries_water.loc['BS325']
            industries_water.loc['BS32610'] = 0.83 * industries_water.loc['BS326']
            industries_water.loc['BS32620'] = 0.17 * industries_water.loc['BS326']
            industries_water.loc['BS327A0'] = 0.38 * industries_water.loc['BS327']
            industries_water.loc['BS32730'] = 0.62 * industries_water.loc['BS327']
            industries_water.loc['BS33100'] = industries_water.loc['BS331']
            industries_water.loc['BS33200'] = industries_water.loc['BS332']
            industries_water.loc['BS33300'] = industries_water.loc['BS333']
            industries_water.loc['BS33410'] = 0.05 * industries_water.loc['BS334']
            industries_water.loc['BS334B0'] = 0.95 * industries_water.loc['BS334']
            industries_water.loc['BS335A0'] = 0.95 * industries_water.loc['BS335']
            industries_water.loc['BS33520'] = 0.05 * industries_water.loc['BS335']
            industries_water.loc['BS33610'] = 0.50 * industries_water.loc['BS336']
            industries_water.loc['BS33620'] = 0.02 * industries_water.loc['BS336']
            industries_water.loc['BS33630'] = 0.24 * industries_water.loc['BS336']
            industries_water.loc['BS33640'] = 0.17 * industries_water.loc['BS336']
            industries_water.loc['BS33650'] = 0.02 * industries_water.loc['BS336']
            industries_water.loc['BS33660'] = 0.02 * industries_water.loc['BS336']
            industries_water.loc['BS33690'] = 0.04 * industries_water.loc['BS336']
            industries_water.loc['BS33700'] = industries_water.loc['BS337']
            industries_water.loc['BS33900'] = industries_water.loc['BS339']
            industries_water.loc['BS4AA00'] = 0.99 * industries_water.loc['BS4A000']
            industries_water.loc['BS453BL'] = 0 * industries_water.loc['BS4A000']
            industries_water.loc['BS453BU'] = 0.01 * industries_water.loc['BS4A000']

            industries_water = industries_water.loc[IOIC_codes]
        elif self.level_of_detail == 'Link-1997 level':

            key_changes = dict.fromkeys(industries_water.index, '')
            for key in key_changes:
                if key + '0' in IOIC_codes:
                    key_changes[key] = key + '0'

            industries_water.loc['BS111A00'] = 0.81 * industries_water.loc['BS111']
            industries_water.loc['BS1114A0'] = 0.08 * industries_water.loc['BS111']
            industries_water.loc['BS111CL0'] = 0.01 * industries_water.loc['BS111']
            industries_water.loc['BS111CU0'] = 0.1 * industries_water.loc['BS111']
            industries_water.loc['BS112000'] = industries_water.loc['BS112']
            industries_water.loc['BS212210'] = 0.15 * industries_water.loc['BS21220']
            industries_water.loc['BS212220'] = 0.35 * industries_water.loc['BS21220']
            industries_water.loc['BS212230'] = 0.4 * industries_water.loc['BS21220']
            industries_water.loc['BS212290'] = 0.1 * industries_water.loc['BS21220']
            industries_water.loc['BS212310'] = 0.16 * industries_water.loc['BS21230']
            industries_water.loc['BS212320'] = 0.15 * industries_water.loc['BS21230']
            industries_water.loc['BS212392'] = 0.18 * industries_water.loc['BS21230']
            industries_water.loc['BS21239A'] = 0.13 * industries_water.loc['BS21230']
            industries_water.loc['BS212396'] = 0.38 * industries_water.loc['BS21230']
            industries_water.loc['BS221200'] = 0.9 * industries_water.loc['BS221A0']
            industries_water.loc['BS221300'] = 0.1 * industries_water.loc['BS221A0']
            industries_water.loc['BS311100'] = 0.08 * industries_water.loc['BS311']
            industries_water.loc['BS311200'] = 0.11 * industries_water.loc['BS311']
            industries_water.loc['BS311300'] = 0.04 * industries_water.loc['BS311']
            industries_water.loc['BS311400'] = 0.07 * industries_water.loc['BS311']
            industries_water.loc['BS311500'] = 0.15 * industries_water.loc['BS311']
            industries_water.loc['BS311600'] = 0.28 * industries_water.loc['BS311']
            industries_water.loc['BS311700'] = 0.06 * industries_water.loc['BS311']
            industries_water.loc['BS311800'] = 0.10 * industries_water.loc['BS311']
            industries_water.loc['BS311900'] = 0.10 * industries_water.loc['BS311']
            industries_water.loc['BS312110'] = 0.30 * industries_water.loc['BS312']
            industries_water.loc['BS312120'] = 0.40 * industries_water.loc['BS312']
            industries_water.loc['BS3121A0'] = 0.16 * industries_water.loc['BS312']
            industries_water.loc['BS312200'] = 0.14 * industries_water.loc['BS312']
            industries_water.loc['BS31A000'] = industries_water.loc['BS31A']
            industries_water.loc['BS31B000'] = industries_water.loc['BS31B']
            industries_water.loc['BS321100'] = 0.52 * industries_water.loc['BS321']
            industries_water.loc['BS321200'] = 0.21 * industries_water.loc['BS321']
            industries_water.loc['BS321900'] = 0.27 * industries_water.loc['BS321']
            industries_water.loc['BS322100'] = 0.62 * industries_water.loc['BS322']
            industries_water.loc['BS322200'] = 0.38 * industries_water.loc['BS322']
            industries_water.loc['BS323000'] = industries_water.loc['BS323']
            industries_water.loc['BS324000'] = industries_water.loc['BS324']
            industries_water.loc['BS325100'] = 0.27 * industries_water.loc['BS325']
            industries_water.loc['BS325B00'] = 0.22 * industries_water.loc['BS325']
            industries_water.loc['BS325300'] = 0.10 * industries_water.loc['BS325']
            industries_water.loc['BS325400'] = 0.23 * industries_water.loc['BS325']
            industries_water.loc['BS325600'] = 0.08 * industries_water.loc['BS325']
            industries_water.loc['BS325900'] = 0.09 * industries_water.loc['BS325']
            industries_water.loc['BS326100'] = 0.83 * industries_water.loc['BS326']
            industries_water.loc['BS326200'] = 0.17 * industries_water.loc['BS326']
            industries_water.loc['BS327A00'] = 0.38 * industries_water.loc['BS327']
            industries_water.loc['BS327300'] = 0.62 * industries_water.loc['BS327']
            industries_water.loc['BS331100'] = 0.17 * industries_water.loc['BS331']
            industries_water.loc['BS331200'] = 0.07 * industries_water.loc['BS331']
            industries_water.loc['BS331300'] = 0.18 * industries_water.loc['BS331']
            industries_water.loc['BS331400'] = 0.54 * industries_water.loc['BS331']
            industries_water.loc['BS331500'] = 0.04 * industries_water.loc['BS331']
            industries_water.loc['BS332100'] = 0.04 * industries_water.loc['BS332']
            industries_water.loc['BS332A00'] = 0.16 * industries_water.loc['BS332']
            industries_water.loc['BS332300'] = 0.41 * industries_water.loc['BS332']
            industries_water.loc['BS332400'] = 0.10 * industries_water.loc['BS332']
            industries_water.loc['BS332500'] = 0.05 * industries_water.loc['BS332']
            industries_water.loc['BS332600'] = 0.03 * industries_water.loc['BS332']
            industries_water.loc['BS332700'] = 0.16 * industries_water.loc['BS332']
            industries_water.loc['BS332800'] = 0.06 * industries_water.loc['BS332']
            industries_water.loc['BS333100'] = 0.25 * industries_water.loc['BS333']
            industries_water.loc['BS333A00'] = 0.26 * industries_water.loc['BS333']
            industries_water.loc['BS333400'] = 0.10 * industries_water.loc['BS333']
            industries_water.loc['BS333500'] = 0.12 * industries_water.loc['BS333']
            industries_water.loc['BS333600'] = 0.04 * industries_water.loc['BS333']
            industries_water.loc['BS333900'] = 0.22 * industries_water.loc['BS333']
            industries_water.loc['BS334100'] = 0.04 * industries_water.loc['BS334']
            industries_water.loc['BS334200'] = 0.24 * industries_water.loc['BS334']
            industries_water.loc['BS334A00'] = 0.44 * industries_water.loc['BS334']
            industries_water.loc['BS334400'] = 0.27 * industries_water.loc['BS334']
            industries_water.loc['BS335100'] = 0.11 * industries_water.loc['BS335']
            industries_water.loc['BS335200'] = 0.05 * industries_water.loc['BS335']
            industries_water.loc['BS335300'] = 0.46 * industries_water.loc['BS335']
            industries_water.loc['BS335900'] = 0.37 * industries_water.loc['BS335']
            industries_water.loc['BS336100'] = 0.50 * industries_water.loc['BS336']
            industries_water.loc['BS336200'] = 0.02 * industries_water.loc['BS336']
            industries_water.loc['BS336300'] = 0.24 * industries_water.loc['BS336']
            industries_water.loc['BS336400'] = 0.17 * industries_water.loc['BS336']
            industries_water.loc['BS336500'] = 0.02 * industries_water.loc['BS336']
            industries_water.loc['BS336600'] = 0.02 * industries_water.loc['BS336']
            industries_water.loc['BS336900'] = 0.04 * industries_water.loc['BS336']
            industries_water.loc['BS337100'] = 0.54 * industries_water.loc['BS337']
            industries_water.loc['BS337200'] = 0.36 * industries_water.loc['BS337']
            industries_water.loc['BS337900'] = 0.10 * industries_water.loc['BS337']
            industries_water.loc['BS339100'] = 0.28 * industries_water.loc['BS339']
            industries_water.loc['BS339900'] = 0.72 * industries_water.loc['BS339']
            industries_water.loc['BS4AA000'] = 0.99 * industries_water.loc['BS4A000']
            industries_water.loc['BS453BL0'] = 0 * industries_water.loc['BS4A000']
            industries_water.loc['BS453BU0'] = 0.01 * industries_water.loc['BS4A000']
            industries_water.loc['BS485100'] = 0.11 * industries_water.loc['BS48B00']
            industries_water.loc['BS48A000'] = 0.09 * industries_water.loc['BS48B00']
            industries_water.loc['BS485300'] = 0.06 * industries_water.loc['BS48B00']
            industries_water.loc['BS488000'] = 0.75 * industries_water.loc['BS48B00']
            industries_water.loc['BS486A00'] = 0.51 * industries_water.loc['BS48600']
            industries_water.loc['BS486200'] = 0.49 * industries_water.loc['BS48600']
            industries_water.loc['BS5121A0'] = 0.77 * industries_water.loc['BS51200']
            industries_water.loc['BS512130'] = 0.14 * industries_water.loc['BS51200']
            industries_water.loc['BS512200'] = 0.08 * industries_water.loc['BS51200']
            industries_water.loc['BS511100'] = 0.09 * industries_water.loc['BS51B00']
            industries_water.loc['BS511200'] = 0.11 * industries_water.loc['BS51B00']
            industries_water.loc['BS51A000'] = 0.74 * industries_water.loc['BS51B00']
            industries_water.loc['BS518000'] = 0.06 * industries_water.loc['BS51B00']
            industries_water.loc['BS521000'] = 0.002 * industries_water.loc['BS52B00']
            industries_water.loc['BS5221A0'] = 0.469 * industries_water.loc['BS52B00']
            industries_water.loc['BS522130'] = 0.042 * industries_water.loc['BS52B00']
            industries_water.loc['BS522A00'] = 0.099 * industries_water.loc['BS52B00']
            industries_water.loc['BS52A000'] = 0.306 * industries_water.loc['BS52B00']
            industries_water.loc['BS524200'] = 0.082 * industries_water.loc['BS52B00']
            industries_water.loc['BS532100'] = 0.28 * industries_water.loc['BS53B00']
            industries_water.loc['BS53A000'] = 0.72 * industries_water.loc['BS53B00']
            industries_water.loc['BS531A00'] = 0.62 * industries_water.loc['BS5A000']
            industries_water.loc['BS551113'] = 0.38 * industries_water.loc['BS5A000']
            industries_water.loc['BS541100'] = 0.26 * industries_water.loc['BS541C0']
            industries_water.loc['BS541200'] = 0.26 * industries_water.loc['BS541C0']
            industries_water.loc['BS541300'] = 0.48 * industries_water.loc['BS541C0']
            industries_water.loc['BS541400'] = 0.03 * industries_water.loc['BS541D0']
            industries_water.loc['BS541500'] = 0.52 * industries_water.loc['BS541D0']
            industries_water.loc['BS541600'] = 0.21 * industries_water.loc['BS541D0']
            industries_water.loc['BS541700'] = 0.08 * industries_water.loc['BS541D0']
            industries_water.loc['BS541900'] = 0.16 * industries_water.loc['BS541D0']
            industries_water.loc['BS561100'] = 0.18 * industries_water.loc['BS56100']
            industries_water.loc['BS561A00'] = 0.16 * industries_water.loc['BS56100']
            industries_water.loc['BS561300'] = 0.17 * industries_water.loc['BS56100']
            industries_water.loc['BS561400'] = 0.11 * industries_water.loc['BS56100']
            industries_water.loc['BS561500'] = 0.07 * industries_water.loc['BS56100']
            industries_water.loc['BS561600'] = 0.09 * industries_water.loc['BS56100']
            industries_water.loc['BS561700'] = 0.23 * industries_water.loc['BS56100']
            industries_water.loc['BS621100'] = 0.44 * industries_water.loc['BS62000']
            industries_water.loc['BS621200'] = 0.21 * industries_water.loc['BS62000']
            industries_water.loc['BS621A00'] = 0.17 * industries_water.loc['BS62000']
            industries_water.loc['BS623000'] = 0.11 * industries_water.loc['BS62000']
            industries_water.loc['BS624000'] = 0.07 * industries_water.loc['BS62000']
            industries_water.loc['BS71A000'] = 0.37 * industries_water.loc['BS71000']
            industries_water.loc['BS713A00'] = 0.37 * industries_water.loc['BS71000']
            industries_water.loc['BS713200'] = 0.26 * industries_water.loc['BS71000']
            industries_water.loc['BS721100'] = 0.19 * industries_water.loc['BS72000']
            industries_water.loc['BS721A00'] = 0.03 * industries_water.loc['BS72000']
            industries_water.loc['BS722000'] = 0.78 * industries_water.loc['BS72000']
            industries_water.loc['BS811100'] = 0.53 * industries_water.loc['BS81100']
            industries_water.loc['BS811A00'] = 0.47 * industries_water.loc['BS81100']
            industries_water.loc['BS812A00'] = 0.59 * industries_water.loc['BS81A00']
            industries_water.loc['BS812200'] = 0.11 * industries_water.loc['BS81A00']
            industries_water.loc['BS812300'] = 0.12 * industries_water.loc['BS81A00']
            industries_water.loc['BS814000'] = 0.18 * industries_water.loc['BS81A00']
            industries_water.loc['GS611100'] = 0.83 * industries_water.loc['GS611B0']
            industries_water.loc['GS611200'] = 0.16 * industries_water.loc['GS611B0']
            industries_water.loc['GS611A00'] = 0.01 * industries_water.loc['GS611B0']
            industries_water.loc['GS911100'] = 0.27 * industries_water.loc['GS91100']
            industries_water.loc['GS911A00'] = 0.73 * industries_water.loc['GS91100']

            new_index = []
            for code in industries_water.index:
                if code in key_changes:
                    if key_changes[code] != '':
                        new_index.append(key_changes[code])
                    else:
                        new_index.append(code)
                else:
                    new_index.append(code)

            industries_water.index = new_index
            # add sectors for which water accounts are zero
            industries_water = pd.concat([industries_water,
                                          pd.Series(0, [i for i in IOIC_codes if i not in industries_water.index])])
            industries_water = industries_water.loc[IOIC_codes]
        elif self.level_of_detail == 'Detail level':

            key_changes = dict.fromkeys(industries_water.index, '')
            for key in key_changes:
                if key + '0' in IOIC_codes:
                    key_changes[key] = key + '0'

            industries_water.loc['BS111A00'] = 0.81 * industries_water.loc['BS111']
            industries_water.loc['BS1114A0'] = 0.08 * industries_water.loc['BS111']
            industries_water.loc['BS111CL0'] = 0.01 * industries_water.loc['BS111']
            industries_water.loc['BS111CU0'] = 0.1 * industries_water.loc['BS111']
            industries_water.loc['BS112A00'] = 0.95 * industries_water.loc['BS112']
            industries_water.loc['BS112500'] = 0.05 * industries_water.loc['BS112']
            industries_water.loc['BS211110'] = 0.5 * industries_water.loc['BS21100']
            industries_water.loc['BS211140'] = 0.5 * industries_water.loc['BS21100']
            industries_water.loc['BS212210'] = 0.15 * industries_water.loc['BS21220']
            industries_water.loc['BS212220'] = 0.35 * industries_water.loc['BS21220']
            industries_water.loc['BS212230'] = 0.4 * industries_water.loc['BS21220']
            industries_water.loc['BS212290'] = 0.1 * industries_water.loc['BS21220']
            industries_water.loc['BS212310'] = 0.16 * industries_water.loc['BS21230']
            industries_water.loc['BS212320'] = 0.15 * industries_water.loc['BS21230']
            industries_water.loc['BS212392'] = 0.18 * industries_water.loc['BS21230']
            industries_water.loc['BS21239A'] = 0.13 * industries_water.loc['BS21230']
            industries_water.loc['BS212396'] = 0.38 * industries_water.loc['BS21230']
            industries_water.loc['BS21311A'] = 0.73 * industries_water.loc['BS21300']
            industries_water.loc['BS21311B'] = 0.27 * industries_water.loc['BS21300']
            industries_water.loc['BS221200'] = 0.9 * industries_water.loc['BS221A0']
            industries_water.loc['BS221300'] = 0.1 * industries_water.loc['BS221A0']
            industries_water.loc['BS311100'] = 0.08 * industries_water.loc['BS311']
            industries_water.loc['BS311200'] = 0.11 * industries_water.loc['BS311']
            industries_water.loc['BS311300'] = 0.04 * industries_water.loc['BS311']
            industries_water.loc['BS311400'] = 0.07 * industries_water.loc['BS311']
            industries_water.loc['BS311500'] = 0.15 * industries_water.loc['BS311']
            industries_water.loc['BS311600'] = 0.28 * industries_water.loc['BS311']
            industries_water.loc['BS311700'] = 0.06 * industries_water.loc['BS311']
            industries_water.loc['BS311800'] = 0.10 * industries_water.loc['BS311']
            industries_water.loc['BS311900'] = 0.10 * industries_water.loc['BS311']
            industries_water.loc['BS312110'] = 0.30 * industries_water.loc['BS312']
            industries_water.loc['BS312120'] = 0.40 * industries_water.loc['BS312']
            industries_water.loc['BS3121A0'] = 0.16 * industries_water.loc['BS312']
            industries_water.loc['BS312200'] = 0.14 * industries_water.loc['BS312']
            industries_water.loc['BS31A000'] = industries_water.loc['BS31A']
            industries_water.loc['BS31B000'] = industries_water.loc['BS31B']
            industries_water.loc['BS321100'] = 0.52 * industries_water.loc['BS321']
            industries_water.loc['BS321200'] = 0.21 * industries_water.loc['BS321']
            industries_water.loc['BS321900'] = 0.27 * industries_water.loc['BS321']
            industries_water.loc['BS322100'] = 0.62 * industries_water.loc['BS322']
            industries_water.loc['BS322200'] = 0.38 * industries_water.loc['BS322']
            industries_water.loc['BS323000'] = industries_water.loc['BS323']
            industries_water.loc['BS324110'] = 0.91 * industries_water.loc['BS324']
            industries_water.loc['BS3241A0'] = 0.09 * industries_water.loc['BS324']
            industries_water.loc['BS325100'] = 0.27 * industries_water.loc['BS325']
            industries_water.loc['BS325200'] = 0.17 * industries_water.loc['BS325']
            industries_water.loc['BS325300'] = 0.10 * industries_water.loc['BS325']
            industries_water.loc['BS325400'] = 0.23 * industries_water.loc['BS325']
            industries_water.loc['BS325500'] = 0.05 * industries_water.loc['BS325']
            industries_water.loc['BS325600'] = 0.08 * industries_water.loc['BS325']
            industries_water.loc['BS325900'] = 0.09 * industries_water.loc['BS325']
            industries_water.loc['BS326100'] = 0.83 * industries_water.loc['BS326']
            industries_water.loc['BS326200'] = 0.17 * industries_water.loc['BS326']
            industries_water.loc['BS327A00'] = 0.38 * industries_water.loc['BS327']
            industries_water.loc['BS327300'] = 0.62 * industries_water.loc['BS327']
            industries_water.loc['BS331100'] = 0.17 * industries_water.loc['BS331']
            industries_water.loc['BS331200'] = 0.07 * industries_water.loc['BS331']
            industries_water.loc['BS331300'] = 0.18 * industries_water.loc['BS331']
            industries_water.loc['BS331400'] = 0.54 * industries_water.loc['BS331']
            industries_water.loc['BS331500'] = 0.04 * industries_water.loc['BS331']
            industries_water.loc['BS332100'] = 0.04 * industries_water.loc['BS332']
            industries_water.loc['BS332A00'] = 0.16 * industries_water.loc['BS332']
            industries_water.loc['BS332300'] = 0.41 * industries_water.loc['BS332']
            industries_water.loc['BS332400'] = 0.10 * industries_water.loc['BS332']
            industries_water.loc['BS332500'] = 0.05 * industries_water.loc['BS332']
            industries_water.loc['BS332600'] = 0.03 * industries_water.loc['BS332']
            industries_water.loc['BS332700'] = 0.16 * industries_water.loc['BS332']
            industries_water.loc['BS332800'] = 0.06 * industries_water.loc['BS332']
            industries_water.loc['BS333100'] = 0.25 * industries_water.loc['BS333']
            industries_water.loc['BS333200'] = 0.12 * industries_water.loc['BS333']
            industries_water.loc['BS333300'] = 0.14 * industries_water.loc['BS333']
            industries_water.loc['BS333400'] = 0.10 * industries_water.loc['BS333']
            industries_water.loc['BS333500'] = 0.12 * industries_water.loc['BS333']
            industries_water.loc['BS333600'] = 0.04 * industries_water.loc['BS333']
            industries_water.loc['BS333900'] = 0.22 * industries_water.loc['BS333']
            industries_water.loc['BS334100'] = 0.04 * industries_water.loc['BS334']
            industries_water.loc['BS334200'] = 0.24 * industries_water.loc['BS334']
            industries_water.loc['BS334A00'] = 0.44 * industries_water.loc['BS334']
            industries_water.loc['BS334400'] = 0.27 * industries_water.loc['BS334']
            industries_water.loc['BS335100'] = 0.11 * industries_water.loc['BS335']
            industries_water.loc['BS335200'] = 0.05 * industries_water.loc['BS335']
            industries_water.loc['BS335300'] = 0.46 * industries_water.loc['BS335']
            industries_water.loc['BS335900'] = 0.37 * industries_water.loc['BS335']
            industries_water.loc['BS336110'] = 0.475 * industries_water.loc['BS336']
            industries_water.loc['BS336120'] = 0.021 * industries_water.loc['BS336']
            industries_water.loc['BS336200'] = 0.024 * industries_water.loc['BS336']
            industries_water.loc['BS336310'] = 0.040 * industries_water.loc['BS336']
            industries_water.loc['BS336320'] = 0.011 * industries_water.loc['BS336']
            industries_water.loc['BS336330'] = 0.016 * industries_water.loc['BS336']
            industries_water.loc['BS336340'] = 0.005 * industries_water.loc['BS336']
            industries_water.loc['BS336350'] = 0.032 * industries_water.loc['BS336']
            industries_water.loc['BS336360'] = 0.044 * industries_water.loc['BS336']
            industries_water.loc['BS336370'] = 0.049 * industries_water.loc['BS336']
            industries_water.loc['BS336390'] = 0.040 * industries_water.loc['BS336']
            industries_water.loc['BS336400'] = 0.170 * industries_water.loc['BS336']
            industries_water.loc['BS336500'] = 0.015 * industries_water.loc['BS336']
            industries_water.loc['BS336600'] = 0.015 * industries_water.loc['BS336']
            industries_water.loc['BS336900'] = 0.042 * industries_water.loc['BS336']
            industries_water.loc['BS337100'] = 0.54 * industries_water.loc['BS337']
            industries_water.loc['BS337200'] = 0.36 * industries_water.loc['BS337']
            industries_water.loc['BS337900'] = 0.10 * industries_water.loc['BS337']
            industries_water.loc['BS339100'] = 0.28 * industries_water.loc['BS339']
            industries_water.loc['BS339900'] = 0.72 * industries_water.loc['BS339']
            industries_water.loc['BS411000'] = 0.02 * industries_water.loc['BS41000']
            industries_water.loc['BS412000'] = 0.04 * industries_water.loc['BS41000']
            industries_water.loc['BS413000'] = 0.13 * industries_water.loc['BS41000']
            industries_water.loc['BS414000'] = 0.17 * industries_water.loc['BS41000']
            industries_water.loc['BS415000'] = 0.12 * industries_water.loc['BS41000']
            industries_water.loc['BS416000'] = 0.14 * industries_water.loc['BS41000']
            industries_water.loc['BS417000'] = 0.23 * industries_water.loc['BS41000']
            industries_water.loc['BS418000'] = 0.12 * industries_water.loc['BS41000']
            industries_water.loc['BS419000'] = 0.03 * industries_water.loc['BS41000']
            industries_water.loc['BS441000'] = 0.154 * industries_water.loc['BS4A000']
            industries_water.loc['BS442000'] = 0.050 * industries_water.loc['BS4A000']
            industries_water.loc['BS443000'] = 0.033 * industries_water.loc['BS4A000']
            industries_water.loc['BS444000'] = 0.076 * industries_water.loc['BS4A000']
            industries_water.loc['BS445000'] = 0.186 * industries_water.loc['BS4A000']
            industries_water.loc['BS446000'] = 0.108 * industries_water.loc['BS4A000']
            industries_water.loc['BS447000'] = 0.060 * industries_water.loc['BS4A000']
            industries_water.loc['BS448000'] = 0.107 * industries_water.loc['BS4A000']
            industries_water.loc['BS451000'] = 0.032 * industries_water.loc['BS4A000']
            industries_water.loc['BS452000'] = 0.101 * industries_water.loc['BS4A000']
            industries_water.loc['BS453A00'] = 0.043 * industries_water.loc['BS4A000']
            industries_water.loc['BS453BL0'] = 0 * industries_water.loc['BS4A000']
            industries_water.loc['BS453BU0'] = 0.010 * industries_water.loc['BS4A000']
            industries_water.loc['BS454000'] = 0.039 * industries_water.loc['BS4A000']
            industries_water.loc['BS485100'] = 0.11 * industries_water.loc['BS48B00']
            industries_water.loc['BS48A000'] = 0.09 * industries_water.loc['BS48B00']
            industries_water.loc['BS485300'] = 0.06 * industries_water.loc['BS48B00']
            industries_water.loc['BS488000'] = 0.75 * industries_water.loc['BS48B00']
            industries_water.loc['BS486A00'] = 0.51 * industries_water.loc['BS48600']
            industries_water.loc['BS486200'] = 0.49 * industries_water.loc['BS48600']
            industries_water.loc['BS491000'] = 0.34 * industries_water.loc['BS49A00']
            industries_water.loc['BS492000'] = 0.66 * industries_water.loc['BS49A00']
            industries_water.loc['BS5121A0'] = 0.77 * industries_water.loc['BS51200']
            industries_water.loc['BS512130'] = 0.14 * industries_water.loc['BS51200']
            industries_water.loc['BS512200'] = 0.08 * industries_water.loc['BS51200']
            industries_water.loc['BS511110'] = 0.04 * industries_water.loc['BS51B00']
            industries_water.loc['BS5111A0'] = 0.05 * industries_water.loc['BS51B00']
            industries_water.loc['BS511200'] = 0.11 * industries_water.loc['BS51B00']
            industries_water.loc['BS515200'] = 0.05 * industries_water.loc['BS51B00']
            industries_water.loc['BS517000'] = 0.66 * industries_water.loc['BS51B00']
            industries_water.loc['BS518000'] = 0.06 * industries_water.loc['BS51B00']
            industries_water.loc['BS519000'] = 0.04 * industries_water.loc['BS51B00']
            industries_water.loc['BS521000'] = 0.002 * industries_water.loc['BS52B00']
            industries_water.loc['BS5221A0'] = 0.469 * industries_water.loc['BS52B00']
            industries_water.loc['BS522130'] = 0.042 * industries_water.loc['BS52B00']
            industries_water.loc['BS522200'] = 0.067 * industries_water.loc['BS52B00']
            industries_water.loc['BS522300'] = 0.031 * industries_water.loc['BS52B00']
            industries_water.loc['BS52A000'] = 0.306 * industries_water.loc['BS52B00']
            industries_water.loc['BS524200'] = 0.082 * industries_water.loc['BS52B00']
            industries_water.loc['BS532100'] = 0.28 * industries_water.loc['BS53B00']
            industries_water.loc['BS532A00'] = 0.56 * industries_water.loc['BS53B00']
            industries_water.loc['BS533000'] = 0.16 * industries_water.loc['BS53B00']
            industries_water.loc['BS531A00'] = 0.62 * industries_water.loc['BS5A000']
            industries_water.loc['BS551113'] = 0.38 * industries_water.loc['BS5A000']
            industries_water.loc['BS541100'] = 0.26 * industries_water.loc['BS541C0']
            industries_water.loc['BS541200'] = 0.26 * industries_water.loc['BS541C0']
            industries_water.loc['BS541300'] = 0.48 * industries_water.loc['BS541C0']
            industries_water.loc['BS541400'] = 0.03 * industries_water.loc['BS541D0']
            industries_water.loc['BS541500'] = 0.52 * industries_water.loc['BS541D0']
            industries_water.loc['BS541600'] = 0.21 * industries_water.loc['BS541D0']
            industries_water.loc['BS541700'] = 0.08 * industries_water.loc['BS541D0']
            industries_water.loc['BS541900'] = 0.16 * industries_water.loc['BS541D0']
            industries_water.loc['BS561100'] = 0.18 * industries_water.loc['BS56100']
            industries_water.loc['BS561A00'] = 0.16 * industries_water.loc['BS56100']
            industries_water.loc['BS561300'] = 0.17 * industries_water.loc['BS56100']
            industries_water.loc['BS561400'] = 0.11 * industries_water.loc['BS56100']
            industries_water.loc['BS561500'] = 0.07 * industries_water.loc['BS56100']
            industries_water.loc['BS561600'] = 0.09 * industries_water.loc['BS56100']
            industries_water.loc['BS561700'] = 0.23 * industries_water.loc['BS56100']
            industries_water.loc['BS621100'] = 0.44 * industries_water.loc['BS62000']
            industries_water.loc['BS621200'] = 0.21 * industries_water.loc['BS62000']
            industries_water.loc['BS621A00'] = 0.17 * industries_water.loc['BS62000']
            industries_water.loc['BS623000'] = 0.11 * industries_water.loc['BS62000']
            industries_water.loc['BS624000'] = 0.07 * industries_water.loc['BS62000']
            industries_water.loc['BS71A000'] = 0.37 * industries_water.loc['BS71000']
            industries_water.loc['BS713A00'] = 0.37 * industries_water.loc['BS71000']
            industries_water.loc['BS713200'] = 0.26 * industries_water.loc['BS71000']
            industries_water.loc['BS721100'] = 0.19 * industries_water.loc['BS72000']
            industries_water.loc['BS721A00'] = 0.03 * industries_water.loc['BS72000']
            industries_water.loc['BS722000'] = 0.78 * industries_water.loc['BS72000']
            industries_water.loc['BS811100'] = 0.53 * industries_water.loc['BS81100']
            industries_water.loc['BS811A00'] = 0.47 * industries_water.loc['BS81100']
            industries_water.loc['BS812A00'] = 0.59 * industries_water.loc['BS81A00']
            industries_water.loc['BS812200'] = 0.11 * industries_water.loc['BS81A00']
            industries_water.loc['BS812300'] = 0.12 * industries_water.loc['BS81A00']
            industries_water.loc['BS814000'] = 0.18 * industries_water.loc['BS81A00']
            industries_water.loc['NP621000'] = 0.09 * industries_water.loc['NPA0000']
            industries_water.loc['NP813A00'] = 0.65 * industries_water.loc['NPA0000']
            industries_water.loc['NP999999'] = 0.26 * industries_water.loc['NPA0000']
            industries_water.loc['GS611100'] = 0.83 * industries_water.loc['GS611B0']
            industries_water.loc['GS611200'] = 0.16 * industries_water.loc['GS611B0']
            industries_water.loc['GS611A00'] = 0.01 * industries_water.loc['GS611B0']
            industries_water.loc['GS911100'] = 0.27 * industries_water.loc['GS91100']
            industries_water.loc['GS911A00'] = 0.73 * industries_water.loc['GS91100']

            new_index = []
            for code in industries_water.index:
                if code in key_changes:
                    if key_changes[code] != '':
                        new_index.append(key_changes[code])
                    else:
                        new_index.append(code)
                else:
                    new_index.append(code)

            industries_water.index = new_index
            # add sectors for which water accounts are zero
            industries_water = pd.concat([industries_water,
                                          pd.Series(0, [i for i in IOIC_codes if i not in industries_water.index])])
            industries_water = industries_water.loc[IOIC_codes]
        elif self.level_of_detail == 'Summary level':
            industries_water.loc['BS11A'] = (industries_water.loc['BS111'] + industries_water.loc['BS112'])
            industries_water.loc['BS113'] = industries_water.loc['BS11300']
            industries_water.loc['BS114'] = industries_water.loc['BS11400']
            industries_water.loc['BS115'] = industries_water.loc['BS11500']
            industries_water.loc['BS210'] = (industries_water.loc['BS21100'] + industries_water.loc['BS21210'] +
                                             industries_water.loc['BS21230'] + industries_water.loc['BS21300'])
            industries_water.loc['BS220'] = (industries_water.loc['BS22110'] + industries_water.loc['BS221A0'])
            industries_water.loc['BS23A'] = industries_water.loc['BS23A00']
            industries_water.loc['BS23B'] = industries_water.loc['BS23B00']
            industries_water.loc['BS23C'] = (industries_water.loc['BS23C10'] + industries_water.loc['BS23C20'] +
                                             industries_water.loc['BS23C30'] + industries_water.loc['BS23C40'] +
                                             industries_water.loc['BS23C50'])
            industries_water.loc['BS23D'] = industries_water.loc['BS23D00']
            industries_water.loc['BS23E'] = industries_water.loc['BS23E00']
            industries_water.loc['BS3A0'] = (industries_water.loc['BS311'] + industries_water.loc['BS312'] +
                                             industries_water.loc['BS31A'] + industries_water.loc['BS31B'] +
                                             industries_water.loc['BS321'] + industries_water.loc['BS322'] +
                                             industries_water.loc['BS323'] + industries_water.loc['BS324'] +
                                             industries_water.loc['BS325'] + industries_water.loc['BS326'] +
                                             industries_water.loc['BS327'] + industries_water.loc['BS331'] +
                                             industries_water.loc['BS332'] + industries_water.loc['BS333'] +
                                             industries_water.loc['BS334'] + industries_water.loc['BS335'] +
                                             industries_water.loc['BS336'] + industries_water.loc['BS337'] +
                                             industries_water.loc['BS339'])
            industries_water.loc['BS410'] = industries_water.loc['BS41000']
            industries_water.loc['BS4A0'] = industries_water.loc['BS4A000']
            industries_water.loc['BS4B0'] = (industries_water.loc['BS48100'] + industries_water.loc['BS48200'] +
                                             industries_water.loc['BS48300'] + industries_water.loc['BS48400'] +
                                             industries_water.loc['BS48B00'] + industries_water.loc['BS48600'] +
                                             industries_water.loc['BS49A00'] + industries_water.loc['BS49300'])
            industries_water.loc['BS510'] = (industries_water.loc['BS51200'] + industries_water.loc['BS51510'] +
                                             industries_water.loc['BS51B00'])
            industries_water.loc['BS53C'] = industries_water.loc['BS5311A']
            industries_water.loc['BS5B0'] = (industries_water.loc['BS52B00'] + industries_water.loc['BS52410'] +
                                             industries_water.loc['BS53110'] + industries_water.loc['BS53B00'] +
                                             industries_water.loc['BS5A000'])
            industries_water.loc['BS540'] = (industries_water.loc['BS541C0'] + industries_water.loc['BS541D0'] +
                                             industries_water.loc['BS54180'])
            industries_water.loc['BS560'] = (industries_water.loc['BS56100'] + industries_water.loc['BS56200'])
            industries_water.loc['BS610'] = industries_water.loc['BS61000']
            industries_water.loc['BS620'] = industries_water.loc['BS62000']
            industries_water.loc['BS710'] = industries_water.loc['BS71000']
            industries_water.loc['BS720'] = industries_water.loc['BS72000']
            industries_water.loc['BS810'] = (industries_water.loc['BS81100'] + industries_water.loc['BS81A00'] +
                                             industries_water.loc['BS81300'])
            industries_water.loc['FC100'] = (industries_water.loc['FC11000'] + industries_water.loc['FC12000'] +
                                             industries_water.loc['FC13000'])
            industries_water.loc['FC200'] = industries_water.loc['FC20000']
            industries_water.loc['FC300'] = industries_water.loc['FC30000']
            industries_water.loc['NP000'] = (industries_water.loc['NP61000'] + industries_water.loc['NP62400'] +
                                             industries_water.loc['NP71000'] + industries_water.loc['NP81310']
                                             + industries_water.loc['NPA0000'])
            industries_water.loc['GS610'] = (industries_water.loc['GS611B0'] + industries_water.loc['GS61130'])
            industries_water.loc['GS620'] = (industries_water.loc['GS62200'] + industries_water.loc['GS62300'])
            industries_water.loc['GS910'] = industries_water.loc['GS91100']
            industries_water.loc['GS920'] = industries_water.loc['GS91200']
            industries_water.loc['GS930'] = industries_water.loc['GS91300']
            industries_water.loc['GS940'] = industries_water.loc['GS91400']

            industries_water = pd.concat([industries_water,
                                          pd.Series(0, [i for i in IOIC_codes if i not in industries_water.index])])
            industries_water = industries_water.loc[IOIC_codes]

        industries_water.index = [i[1] for i in self.industries]
        industries_water.name = ('Water use', '')
        industries_water *= 1000
        self.F = self.F.append(industries_water)

        self.emission_metadata.loc[('Water use', ''), 'CAS Number'] = 'N/A'
        self.emission_metadata.loc[('Water use', ''), 'Unit'] = 'm3'

        # normalize
        if self.classification == 'industry':
            self.S = self.F.dot(self.inv_g)

        if self.classification == 'product':
            self.F = self.F.dot(self.V.dot(self.inv_g).T)
            self.S = self.F.dot(self.inv_q)

    def characterization_matrix(self):

        IW = pd.read_excel(pkg_resources.resource_stream(__name__, '/Data/Impact_World.xlsx'))

        df = IW.set_index('CAS number')
        df.index = [str(i) for i in df.index]
        df = df.groupby(df.index).head(n=1)

        concordance_IW = dict.fromkeys(self.F.index.levels[0])

        for pollutant in concordance_IW:
            match_CAS = ''
            try:
                if len(self.emission_metadata.loc[(pollutant, 'air'), 'CAS Number'].split('-')[0]) == 2:
                    match_CAS = '0000' + self.emission_metadata.loc[(pollutant, 'air'), 'CAS Number']
                elif len(self.emission_metadata.loc[(pollutant, 'air'), 'CAS Number'].split('-')[0]) == 3:
                    match_CAS = '000' + self.emission_metadata.loc[(pollutant, 'air'), 'CAS Number']
                elif len(self.emission_metadata.loc[(pollutant, 'air'), 'CAS Number'].split('-')[0]) == 4:
                    match_CAS = '00' + self.emission_metadata.loc[(pollutant, 'air'), 'CAS Number']
                elif len(self.emission_metadata.loc[(pollutant, 'air'), 'CAS Number'].split('-')[0]) == 5:
                    match_CAS = '0' + self.emission_metadata.loc[(pollutant, 'air'), 'CAS Number']
                elif len(self.emission_metadata.loc[(pollutant, 'air'), 'CAS Number'].split('-')[0]) == 6:
                    match_CAS = self.emission_metadata.loc[(pollutant, 'air'), 'CAS Number']
                try:
                    concordance_IW[pollutant] = [df.loc[i, 'Elem flow name'] for i in df.index if i == match_CAS][0]

                except IndexError:
                    pass
            except KeyError:
                pass

        # hardcoding what could not be matched using CAS number
        concordance_IW['Aluminum oxide (fibrous forms only)'] = 'Aluminium'
        concordance_IW['Antimony (and its compounds)'] = 'Antimony'
        concordance_IW['Arsenic (and its compounds)'] = 'Arsenic'
        concordance_IW['Cadmium (and its compounds)'] = 'Cadmium'
        concordance_IW['Chromium (and its compounds)'] = 'Chromium'
        concordance_IW['Cobalt (and its compounds)'] = 'Cobalt'
        concordance_IW['Copper (and its compounds)'] = 'Copper'
        concordance_IW['Fluorine'] = 'Fluorine'
        concordance_IW['Lead (and its compounds)'] = 'Lead'
        concordance_IW['Nickel (and its compounds)'] = 'Nickel'
        concordance_IW['Molybdenum (and its compounds)'] = 'Molybdenum'
        concordance_IW['Mercury (and its compounds)'] = 'Mercury'
        concordance_IW['Manganese (and its compounds)'] = 'Manganese'
        concordance_IW['PM10 - Particulate matter <=10 microns'] = 'Particulates, < 10 um'
        concordance_IW['PM2.5 - Particulate matter <=2.5 microns'] = 'Particulates, < 2.5 um'
        concordance_IW['Phosphorus (and its compounds)'] = 'Phosphorus'
        concordance_IW['Selenium (and its compounds)'] = 'Selenium'
        concordance_IW['Silver (and its compounds)'] = 'Silver'
        concordance_IW['Thallium (and its compounds)'] = 'Thallium'
        concordance_IW['Zinc (and its compounds)'] = 'Zinc'
        concordance_IW['Speciated VOC - Butane  (all isomers)'] = 'Butane'
        concordance_IW['Speciated VOC - Butene  (all isomers)'] = '1-Butene'
        concordance_IW['Speciated VOC - Anthraquinone (all isomers)'] = 'Anthraquinone'
        concordance_IW['Speciated VOC - Cycloheptane'] = 'Cycloheptane'
        concordance_IW['Speciated VOC - Cyclohexene'] = 'Cyclohexene'
        concordance_IW['Speciated VOC - Cyclooctane'] = 'Cyclooctane'
        concordance_IW['Speciated VOC - Decane  (all isomers)'] = 'Decane'
        concordance_IW['Speciated VOC - Dodecane  (all isomers)'] = 'Dodecane'
        concordance_IW['Speciated VOC - Heptane  (all isomers)'] = 'Heptane'
        concordance_IW['Speciated VOC - Hexane'] = 'Hexane'
        concordance_IW['Speciated VOC - Nonane  (all isomers)'] = 'Nonane'
        concordance_IW['Speciated VOC - Octane  (all isomers)'] = 'N-octane'
        concordance_IW['Speciated VOC - Pentane  (all isomers)'] = 'Pentane'
        concordance_IW['Speciated VOC - Pentene  (all isomers)'] = '1-Pentene'

        pivoting = pd.pivot_table(IW, values='CF value', index='Impact category',
                                  columns=['Elem flow name', 'Compartment', 'Sub-compartment']).fillna(0)

        self.C = pd.DataFrame(0, pivoting.index, self.F.index)
        for flow in self.C.columns:
            if concordance_IW[flow[0]] != None:
                try:
                    self.C.loc[:, (flow[0], flow[1])] = pivoting.loc[:, (concordance_IW[flow[0]], flow[1], '(unspecified)')]
                except KeyError:
                    pass
        self.C.loc['Climate change', ('GHGs', '')] = 1
        self.C.loc['Energy use', ('Energy use', '')] = 1
        self.C.loc['Water use', ('Water use', '')] = 1
        # remove impact categories with no pollutants
        self.C = self.C.loc[self.C.sum(axis=1)[self.C.sum(axis=1) != 0].index].fillna(0)

        self.methods_metadata = IW.loc[:, ['Impact category', 'CF unit']]
        self.methods_metadata = self.methods_metadata.set_index('Impact category')
        self.methods_metadata = self.methods_metadata.groupby(self.methods_metadata.index).head(n=1)
        self.methods_metadata = pd.concat([self.methods_metadata ,
                                           pd.DataFrame(['kgCO2eq', 'TJ', 'm3'],
                                                        ['Climate change', 'Energy use', 'Water use'],
                                                        ['CF unit'])])
        self.methods_metadata = self.methods_metadata.loc[self.C.index]


def extract_data_from_csv(file_name):
    year = 2017
    if file_name == 'Water_use.csv':
        year = 2015
    df = pd.read_csv(pkg_resources.resource_stream(__name__, '/Data/'+file_name))
    df = df.loc[[i for i in df.index if df.loc[i, 'GEO'] == 'Canada' and df.loc[i, 'REF_DATE'] == year]]
    df.set_index('Sector', inplace=True)
    return df


def select_industries_emissions(df):
    industries = df.loc[[i for i in df.index if ('Total' not in i
                                                 and 'Balancing' not in i
                                                 and 'Households' not in i)],
                        'VALUE'].fillna(0)

    industries.index = [i.split('[')[1].split(']')[0] for i in industries.index]
    return industries
