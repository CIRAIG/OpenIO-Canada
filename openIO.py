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
import os
# from pyomo.environ import *
from time import time
import pymrio


class IOTables:
    def __init__(self, folder_path, classification, exiobase_folder=None):
        """
        :param folder_path: [string] the path to the folder with the economic data (e.g. /../Detail level/)
        :param classification: [string] the type of classification to adopt for the symmetric IOT ("product" or "industry")
        :param exiobase_folder: [string] path to exiobase folder for international imports (optional)
        """

        start = time()
        print("Reading all the Excel files...")

        self.level_of_detail = [i for i in folder_path.split('/') if 'level' in i][0]
        self.classification = classification
        self.exiobase_folder = exiobase_folder

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
        self.INT_imports = pd.DataFrame()
        self.F_INT_imports = pd.DataFrame()
        self.S_INT_imports = pd.DataFrame()
        self.E_INT_imports = pd.DataFrame()
        self.C_INT_imports = pd.DataFrame()
        self.L = pd.DataFrame()
        self.E = pd.DataFrame()
        self.D = pd.DataFrame()

        # metadata
        self.emission_metadata = pd.DataFrame()
        self.methods_metadata = pd.DataFrame()
        self.industries = []
        self.commodities = []
        self.factors_of_production = []

        self.matching_dict = {'AB': 'Alberta',
                                'BC': 'British Columbia',
                                'MB': 'Manitoba',
                                'NB': 'New Brunswick',
                                'NL': 'Newfoundland and Labrador',
                                'NS': 'Nova Scotia',
                                'NT': 'Northwest Territories',
                                'NU': 'Nunavut',
                                'ON': 'Ontario',
                                'PE': 'Prince Edward Island',
                                'QC': 'Quebec',
                                'SK': 'Saskatchewan',
                                'YT': 'Yukon'}

        files = [i for i in os.walk(folder_path)]
        files = [i for i in files[0][2] if i[:2] in self.matching_dict.keys() and 'SUT' in i]
        self.year = int(files[0].split('SUT_C')[1].split('_')[0])

        try:
            self.NPRI = pd.read_excel(pkg_resources.resource_stream(__name__, '/Data/NPRI-INRP_DataDonnées_' +
                                                                    str(self.year) + '.xlsx'), None)
        # 2017 by default
        except FileNotFoundError:
            self.NPRI = pd.read_excel(pkg_resources.resource_stream(__name__, '/Data/NPRI-INRP_DataDonnées_2017.xlsx'),
                                      None)

        print("Formatting the Supply and Use tables...")
        for province_data in files:
            su_tables = pd.read_excel(folder_path+province_data, None)
            region = province_data[:2]
            self.format_tables(su_tables, region)

        self.W = self.W.fillna(0)
        self.WY = self.WY.fillna(0)
        self.Y = self.Y.fillna(0)
        self.q = self.q.fillna(0)
        self.g = self.g.fillna(0)
        self.U = self.U.fillna(0)
        self.V = self.V.fillna(0)

        print("Modifying names of duplicated sectors...")
        self.dealing_with_duplicated_names()

        print('Aggregating final demand sectors...')
        self.aggregate_final_demand()

        print('Removing IOIC codes from index...')
        self.remove_codes()

        print("Balancing inter-provincial trade...")
        self.province_import_export(
            pd.read_excel(
                folder_path+[i for i in [j for j in os.walk(folder_path)][0][2] if 'Provincial_trade_flow' in i][0],
                'Data'))

        if self.exiobase_folder:
            print('Linking international imports to Exiobase...')
            self.international_import_export()

        print("Building the symmetric tables...")
        self.gimme_symmetric_iot()

        print("Balancing value added...")
        self.balance_value_added()

        print("Extracting and formatting environmental data from the NPRI file...")
        self.extract_environmental_data()

        print("Matching emission data from NPRI to IOT sectors...")
        self.match_npri_data_to_iots()

        print("Matching GHG accounts to IOT sectors...")
        self.match_ghg_accounts_to_iots()

        print("Matching water accounts to IOT sectors...")
        self.match_water_accounts_to_iots()

        print("Creating the characterization matrix...")
        self.characterization_matrix()

        print("Normalizing emissions...")
        self.normalize_flows()

        print('Took '+str(time()-start)+' seconds')

    def format_tables(self, su_tables, region):
        """
        Extracts the relevant dataframes from the Excel files in the Stat Can folder
        :param su_tables: the supply and use economic tables
        :param region: the province of Canada to compile data for
        :return: self.W, self.WY, self.Y, self.g, self.q, self.V, self.U
        """

        supply_table = su_tables['Supply'].copy()
        use_table = su_tables['Use_Basic'].copy()

        if not self.industries:
            for i in range(0, len(supply_table.columns)):
                if supply_table.iloc[11, i] == 'Total':
                    break
                if supply_table.iloc[11, i] not in [np.nan, 'Industries']:
                    # tuple with code + name (need code to deal with duplicate names in detailed levels)
                    self.industries.append((supply_table.iloc[12, i], supply_table.iloc[11, i]))
            # remove fictive sectors
            self.industries = [i for i in self.industries if not re.search(r'^F', i[0])]

        if not self.commodities:
            for i, element in enumerate(supply_table.iloc[:, 0].tolist()):
                if type(element) == str:
                    # identify by their codes
                    if re.search(r'^[M,F,N,G,I,E]\w*\d', element):
                        self.commodities.append((element, supply_table.iloc[i, 1]))
                    elif re.search(r'^P\w*\d', element) or re.search(r'^GVA', element):
                        self.factors_of_production.append((element, supply_table.iloc[i, 1]))

        final_demand = []
        for i in range(0, len(use_table.columns)):
            if use_table.iloc[11, i] == 'Total use':
                break
            if use_table.iloc[11, i] not in [np.nan, 'Industries']:
                final_demand.append((use_table.iloc[12, i], use_table.iloc[11, i]))
        final_demand = [i for i in final_demand if i not in self.industries and i[1] != 'Total']

        df = supply_table.iloc[14:, 2:]
        df.index = list(zip(supply_table.iloc[14:, 0].tolist(), supply_table.iloc[14:, 1].tolist()))
        df.columns = list(zip(supply_table.iloc[12, 2:].tolist(), supply_table.iloc[11, 2:].tolist()))
        supply_table = df

        df = use_table.iloc[14:, 2:]
        df.index = list(zip(use_table.iloc[14:, 0].tolist(), use_table.iloc[14:, 1].tolist()))
        df.columns = list(zip(use_table.iloc[12, 2:].tolist(), use_table.iloc[11, 2:].tolist()))
        use_table = df

        # fill with zeros
        supply_table.replace('.', 0, inplace=True)
        use_table.replace('.', 0, inplace=True)

        # get strings as floats
        supply_table = supply_table.astype('float64')
        use_table = use_table.astype('float64')

        if self.level_of_detail == 'Detail level':
            # tables from k$ to $
            supply_table *= 1000
            use_table *= 1000
        else:
            # tables from M$ to $
            supply_table *= 1000000
            use_table *= 1000000

        # check calculated totals matched displayed totals
        assert np.allclose(use_table.iloc[:, use_table.columns.get_loc(('TOTAL', 'Total'))],
                           use_table.iloc[:, :use_table.columns.get_loc(('TOTAL', 'Total'))].sum(axis=1), atol=1e-5)
        assert np.allclose(supply_table.iloc[supply_table.index.get_loc(('TOTAL', 'Total'))],
                           supply_table.iloc[:supply_table.index.get_loc(('TOTAL', 'Total'))].sum(), atol=1e-5)

        # extract the tables we need
        W = use_table.loc[self.factors_of_production, self.industries]
        W.drop(('GVA', 'Gross value-added at basic prices'), inplace=True)
        Y = use_table.loc[self.commodities, final_demand]
        WY = use_table.loc[self.factors_of_production, final_demand]
        WY.drop(('GVA', 'Gross value-added at basic prices'), inplace=True)
        g = use_table.loc[[('TOTAL', 'Total')], self.industries]
        q = supply_table.loc[self.commodities, [('TOTAL', 'Total')]]
        V = supply_table.loc[self.commodities, self.industries]
        U = use_table.loc[self.commodities, self.industries]
        INT_imports = supply_table.loc[self.commodities,
                                       [i for i in supply_table.columns if re.search(r'^INTIM', i[0])]]

        # create multiindex with region as first level
        for matrix in [W, Y, WY, g, q, V, U, INT_imports]:
            matrix.columns = pd.MultiIndex.from_product([[region], matrix.columns]).tolist()
            matrix.index = pd.MultiIndex.from_product([[region], matrix.index]).tolist()

        # concat the region tables with the all the other tables
        self.W = pd.concat([self.W, W])
        self.WY = pd.concat([self.WY, WY])
        self.Y = pd.concat([self.Y, Y])
        self.q = pd.concat([self.q, q])
        self.g = pd.concat([self.g, g])
        self.U = pd.concat([self.U, U])
        self.V = pd.concat([self.V, V])
        self.INT_imports = pd.concat([self.INT_imports, INT_imports])

        # assert np.isclose(self.V.sum().sum(), self.g.sum().sum())
        # assert np.isclose(self.U.sum().sum()+self.Y.drop([
        #     i for i in self.Y.columns if i[1] == ('IPTEX', 'Interprovincial exports')], axis=1).sum().sum(),
        #                   self.q.sum().sum())

    def dealing_with_duplicated_names(self):
        """
        IOIC classification has duplicate names, so we rename when it's the case
        :return: updated dataframes
        """

        # reindexing to fix the order of the columns
        self.V = self.V.T.reindex(pd.MultiIndex.from_product([self.matching_dict, self.industries]).tolist()).T
        self.U = self.U.T.reindex(pd.MultiIndex.from_product([self.matching_dict, self.industries]).tolist()).T
        self.g = self.g.T.reindex(pd.MultiIndex.from_product([self.matching_dict, self.industries]).tolist()).T
        self.W = self.W.T.reindex(pd.MultiIndex.from_product([self.matching_dict, self.industries]).tolist()).T

        if self.level_of_detail in ['Link-1961 level', 'Link-1997 level', 'Detail level']:
            self.industries = [(i[0], i[1] + ' (private)') if re.search(r'^BS61', i[0]) else i for i in
                               self.industries]
            self.industries = [(i[0], i[1] + ' (non-profit)') if re.search(r'^NP61|^NP71', i[0]) else i for i in
                               self.industries]
            self.industries = [(i[0], i[1] + ' (public)') if re.search(r'^GS61', i[0]) else i for i in
                               self.industries]
        if self.level_of_detail in ['Link-1997 level', 'Detail level']:
            self.industries = [(i[0], i[1] + ' (private)') if re.search(r'^BS623|^BS624', i[0]) else i for i in
                               self.industries]
            self.industries = [(i[0], i[1] + ' (non-profit)') if re.search(r'^NP624', i[0]) else i for i in
                               self.industries]
            self.industries = [(i[0], i[1] + ' (public)') if re.search(r'^GS623', i[0]) else i for i in
                               self.industries]

        # applying the change of names to columns
        for df in [self.V, self.U, self.g, self.W]:
            df.columns = pd.MultiIndex.from_product([self.matching_dict, self.industries]).tolist()

    def aggregate_final_demand(self):
        """
        Aggregates all final demand sectors into 6 elements: ["Household final consumption expenditure",
        "Non-profit institutions serving households' final consumption expenditure",
        "Governments final consumption expenditure", "Gross fixed capital formation", "Changes in inventories",
        "International exports"]
        Provincial exports will be included in self.U and are thus excluded from self.Y
        :return: self.Y with final demand sectors aggregated
        """

        # final demands are identified through their codes, hence the use of regex
        aggregated_Y = self.Y.loc[:, [i for i in self.Y.columns if
                                      re.search(r'^PEC\w*\d', i[1][0])]].groupby(level=0, axis=1).sum()
        aggregated_Y.columns = pd.MultiIndex.from_product([aggregated_Y.columns,
                                                           ["Household final consumption expenditure"]])

        df = self.Y.loc[:, [i for i in self.Y.columns if
                            re.search(r'^CEN\w*\d', i[1][0])]].groupby(level=0, axis=1).sum()
        df.columns = pd.MultiIndex.from_product([
            df.columns, ["Non-profit institutions serving households' final consumption expenditure"]])
        aggregated_Y = pd.concat([aggregated_Y, df], axis=1)

        df = self.Y.loc[:, [i for i in self.Y.columns if
                            re.search(r'^CEG\w*\d', i[1][0])]].groupby(level=0, axis=1).sum()
        df.columns = pd.MultiIndex.from_product([
            df.columns, ["Governments final consumption expenditure"]])
        aggregated_Y = pd.concat([aggregated_Y, df], axis=1)

        df = self.Y.loc[:, [i for i in self.Y.columns if
                            re.search(r'^CO\w*\d|^ME\w*\d|^IP\w[T]*\d', i[1][0])]].groupby(level=0, axis=1).sum()
        df.columns = pd.MultiIndex.from_product([
            df.columns, ["Gross fixed capital formation"]])
        aggregated_Y = pd.concat([aggregated_Y, df], axis=1)

        df = self.Y.loc[:, [i for i in self.Y.columns if
                            re.search(r'^INV\w*\d', i[1][0])]].groupby(level=0, axis=1).sum()
        df.columns = pd.MultiIndex.from_product([
            df.columns, ["Changes in inventories"]])
        aggregated_Y = pd.concat([aggregated_Y, df], axis=1)

        df = self.Y.loc[:, [i for i in self.Y.columns if
                            re.search(r'^INT\w*', i[1][0])]].groupby(level=0, axis=1).sum()
        df.columns = pd.MultiIndex.from_product([
            df.columns, ["International exports"]])
        aggregated_Y = pd.concat([aggregated_Y, df], axis=1)

        self.Y = aggregated_Y
        self.Y = self.Y.T.sort_index().T

        aggregated_WY = self.WY.loc[:, [i for i in self.WY.columns if
                                      re.search(r'^PEC\w*\d', i[1][0])]].groupby(level=0, axis=1).sum()
        aggregated_WY.columns = pd.MultiIndex.from_product([aggregated_WY.columns,
                                                           ["Household final consumption expenditure"]])

        df = self.WY.loc[:, [i for i in self.WY.columns if
                            re.search(r'^CEN\w*\d', i[1][0])]].groupby(level=0, axis=1).sum()
        df.columns = pd.MultiIndex.from_product([
            df.columns, ["Non-profit institutions serving households' final consumption expenditure"]])
        aggregated_WY = pd.concat([aggregated_WY, df], axis=1)

        df = self.WY.loc[:, [i for i in self.WY.columns if
                            re.search(r'^CEG\w*\d', i[1][0])]].groupby(level=0, axis=1).sum()
        df.columns = pd.MultiIndex.from_product([
            df.columns, ["Governments final consumption expenditure"]])
        aggregated_WY = pd.concat([aggregated_WY, df], axis=1)

        df = self.WY.loc[:, [i for i in self.WY.columns if
                            re.search(r'^CO\w*\d|^ME\w*\d|^IP\w*\d', i[1][0])]].groupby(level=0, axis=1).sum()
        df.columns = pd.MultiIndex.from_product([
            df.columns, ["Gross fixed capital formation"]])
        aggregated_WY = pd.concat([aggregated_WY, df], axis=1)

        df = self.WY.loc[:, [i for i in self.WY.columns if
                            re.search(r'^INV\w*\d', i[1][0])]].groupby(level=0, axis=1).sum()
        df.columns = pd.MultiIndex.from_product([
            df.columns, ["Changes in inventories"]])
        aggregated_WY = pd.concat([aggregated_WY, df], axis=1)

        df = self.WY.loc[:, [i for i in self.WY.columns if
                            re.search(r'^INT\w*', i[1][0])]].groupby(level=0, axis=1).sum()
        df.columns = pd.MultiIndex.from_product([
            df.columns, ["International exports"]])
        aggregated_WY = pd.concat([aggregated_WY, df], axis=1)

        self.WY = aggregated_WY
        self.WY = self.WY.T.sort_index().T

        for df in [self.Y, self.WY]:
            assert len([i for i in df.columns.levels[1] if i not in [
                "Household final consumption expenditure",
                "Non-profit institutions serving households' final consumption expenditure",
                "Governments final consumption expenditure",
                "Gross fixed capital formation",
                "Changes in inventories",
                "International exports"
            ]]) == 0

    def remove_codes(self):
        """
        Removes the IOIC codes from the index to only leave the name.
        :return: Dataframes with the code of the multi-index removed
        """
        # removing the IOIC codes
        for df in [self.W, self.g, self.V, self.U, self.INT_imports]:
            df.columns = [(i[0], i[1][1]) for i in df.columns]
        for df in [self.W, self.Y, self.WY, self.q, self.V, self.U, self.INT_imports]:
            df.index = [(i[0], i[1][1]) for i in df.index]

        # recreating MultiIndexes
        for df in [self.W, self.Y, self.WY, self.g, self.q, self.V, self.U, self.INT_imports]:
            df.index = pd.MultiIndex.from_tuples(df.index)
            df.columns = pd.MultiIndex.from_tuples(df.columns)

        # reordering columns
        reindexed_columns = pd.MultiIndex.from_product([list(self.matching_dict.keys()),
                                                        [i[1] for i in self.industries]])
        self.W = self.W.T.reindex(reindexed_columns).T
        self.g = self.g.T.reindex(reindexed_columns).T
        self.V = self.V.T.reindex(reindexed_columns).T
        self.U = self.U.T.reindex(reindexed_columns).T

    def province_import_export(self, province_trade_file):
        """
        Method extracting and formatting inter province imports/exports
        :return: modified self.U, self.V, self.W, self.Y
        """

        province_trade_file = province_trade_file

        province_trade_file.Origin = [{v: k for k, v in self.matching_dict.items()}[i.split(') ')[1]] if (
                ')' in i and i != '(81) Canadian territorial enclaves abroad') else i for i in
                                      province_trade_file.Origin]
        province_trade_file.Destination = [{v: k for k, v in self.matching_dict.items()}[i.split(') ')[1]] if (
                ')' in i and i != '(81) Canadian territorial enclaves abroad') else i for i in
                                           province_trade_file.Destination]
        # extracting and formatting supply for each province
        province_trade = pd.pivot_table(data=province_trade_file, index='Destination', columns=['Origin', 'Product'])

        province_trade = province_trade.loc[
            [i for i in province_trade.index if i in self.matching_dict], [i for i in province_trade.columns if
                                                                           i[1] in self.matching_dict]]
        if self.level_of_detail == 'Detail level':
            province_trade *= 1000
        else:
            province_trade *= 1000000
        province_trade.columns = [(i[1], i[2].split(': ')[1]) if ':' in i[2] else i for i in
                                  province_trade.columns]
        province_trade.drop([i for i in province_trade.columns if i[1] not in [i[1] for i in self.commodities]],
                            axis=1, inplace=True)
        province_trade.columns = pd.MultiIndex.from_tuples(province_trade.columns)
        for province in province_trade.index:
            province_trade.loc[province, province] = 0

        import_markets = pd.DataFrame(0, province_trade.index, province_trade.columns)
        for importing_province in province_trade.index:
            for exported_product in province_trade.columns.levels[1]:
                import_markets.loc[
                    importing_province, [i for i in import_markets.columns if i[1] == exported_product]] = (
                            province_trade.loc[
                                importing_province, [i for i in province_trade.columns if i[1] == exported_product]] /
                            province_trade.loc[importing_province, [i for i in province_trade.columns if
                                                                    i[1] == exported_product]].sum()).values

        for importing_province in province_trade.index:
            U_Y = pd.concat([self.U.loc[importing_province, importing_province],
                             self.Y.loc[importing_province, importing_province]], axis=1)
            # not distributing interprovincial trade flows to negative values
            U_Y = U_Y[U_Y > 0].fillna(0)
            total_imports = province_trade.groupby(level=1, axis=1).sum().loc[importing_province]
            index_commodity = [i[1] for i in self.commodities]
            total_imports = total_imports.reindex(index_commodity).fillna(0)
            import_distribution = ((U_Y.T / (U_Y.sum(axis=1))) * total_imports).T.fillna(0)

            # distribution balance imports to the different exporting regions
            final_demand_imports = [i for i in import_distribution.columns if i not in self.U.columns.levels[1]]
            for exporting_province in province_trade.index:
                if importing_province != exporting_province:
                    df = (
                        (import_distribution.T * import_markets.fillna(0).loc[
                            importing_province, exporting_province]).T
                    ).reindex(import_distribution.index).fillna(0)
                    # assert index and columns are the same before using .values
                    assert all(self.U.loc[exporting_province, importing_province].index == df.loc[:,
                                                                                           self.U.columns.levels[
                                                                                               1]].reindex(
                        self.U.loc[exporting_province, importing_province].columns, axis=1).index)
                    assert all(self.U.loc[exporting_province, importing_province].columns == df.loc[:,
                                                                                             self.U.columns.levels[
                                                                                                 1]].reindex(
                        self.U.loc[exporting_province, importing_province].columns, axis=1).columns)
                    # assign new values into self.U and self.Y
                    self.U.loc[exporting_province, importing_province] = df.loc[:,
                                                                         self.U.columns.levels[1]].reindex(
                        self.U.loc[exporting_province, importing_province].columns, axis=1).values
                    self.Y.loc[exporting_province, importing_province].update(df.loc[:, final_demand_imports])

            # remove interprovincial from intraprovincial to not double count
            self.U.loc[importing_province, importing_province].update(
                self.U.loc[importing_province, importing_province] - self.U.loc[
                    [i for i in self.matching_dict if i != importing_province], importing_province].groupby(
                    level=1).sum())
            self.Y.loc[importing_province, importing_province].update(
                self.Y.loc[importing_province, importing_province] - self.Y.loc[
                    [i for i in self.matching_dict if i != importing_province], importing_province].groupby(
                    level=1).sum())

            # if some province buys more than they use, drop the value in "changes in inventories"
            # if it occurs, it's probably linked to the immediate re-export to other provinces
            if not len(self.U[self.U > -1].dropna()) == len(self.U):
                product_creating_issue_index = self.U[self.U < -1].dropna(how='all').dropna(1).index
                product_creating_issue_column = self.U[self.U < -1].dropna(how='all').dropna(1).columns
                value_to_balance = self.U[self.U < -1].dropna(how='all').dropna(1).iloc[0, 0]
                self.U.loc[product_creating_issue_index, product_creating_issue_column] = 0
                self.Y.loc[product_creating_issue_index, (product_creating_issue_index[0][0],
                                                          'Changes in inventories')] += - value_to_balance
            # removing negative values lower than 1$ (potential calculation artefacts)
            self.U = self.U[self.U > 0].fillna(0)
            # checking negative values were removed
            assert not self.U[self.U < 0].any().any()

    def international_import_export(self):
        """
        Method executes two things:
        1. It removes international imports from the use table
        2. It estimates the emissions (or the impacts) from these international imports, based on exiobase
        Resulting emissions are stored in self.F_INT_imports
        :returns self.C_INT_imports, self.F_INT_imports, modified self.U
        """

        # 1. Removing international imports

        # aggregating international imports in 1 column
        self.INT_imports = self.INT_imports.groupby(axis=1, level=1).sum()
        # concat U and Y to look at all users (industry + final demand)
        U_Y = pd.concat([self.U, self.Y], axis=1)
        # weighted average of who is requiring the international imports, based on national use
        who_uses_int_imports = (U_Y.T / U_Y.sum(1)).T * self.INT_imports.values
        # remove international imports from national use
        self.U = self.U - who_uses_int_imports.reindex(self.U.columns, axis=1)
        # check no issues of negatives
        assert len(self.U[self.U < -1].dropna(how='all', axis=1).dropna(how='all', axis=0)) == 0
        self.U = self.U[self.U > 0].fillna(0)
        assert not self.U[self.U < 0].any().any()
        # remove international imports from final demand
        self.Y = self.Y - who_uses_int_imports.reindex(self.Y.columns, axis=1)
        # remove negative artefacts (because of negative values in inventories)
        self.Y = pd.concat([self.Y[self.Y >= 0].fillna(0), self.Y[self.Y < -1].fillna(0)], axis=1)
        self.Y = self.Y.groupby(by=self.Y.columns, axis=1).sum()
        self.Y.columns = pd.MultiIndex.from_tuples(self.Y.columns)

        # 2. Estimating the emissions of international imports

        # importing exiobase
        io = pymrio.parse_exiobase3(self.exiobase_folder)
        io.calc_all()

        # selecting the countries which make up the international imports
        INT_countries = [i for i in io.get_regions().tolist() if i != 'CA']

        # importing the concordance between open IO and exiobase classifications
        ioic_exio = pd.read_excel(pkg_resources.resource_stream(__name__, '/Data/IOIC_EXIOBASE.xlsx'),
                                  'commodities')
        # make concordance on codes because Statcan changes names of sectors with updates
        ioic_exio = ioic_exio[2:].drop('IOIC Detail level - EXIOBASE', axis=1).set_index('Unnamed: 1').fillna(0)
        ioic_exio.index.name = None

        norm_emissions_INT_imports = pd.DataFrame(0, io.satellite.S.index, [i[0] for i in self.commodities])
        for product in norm_emissions_INT_imports.columns:
            if len(ioic_exio.loc[product][ioic_exio.loc[product] == 1].index) != 0:
                df = io.x.loc(axis=0)[:, ioic_exio.loc[product][ioic_exio.loc[product] == 1].index]
                df = df.loc[INT_countries] / df.loc[INT_countries].sum()
                norm_emissions_INT_imports.loc[:, product] = (
                    io.satellite.S.reindex(df.index, axis=1).dot(df)).values
        norm_emissions_INT_imports = norm_emissions_INT_imports.fillna(0)
        # exiobase in millions
        norm_emissions_INT_imports[9:] /= 1000000
        # exiobase in euros
        norm_emissions_INT_imports *= 1.5
        # from codes to names
        norm_emissions_INT_imports.columns = [dict(self.commodities)[i] for i in norm_emissions_INT_imports.columns]

        for province in self.matching_dict:
            diag = pd.DataFrame(np.diagflat(self.INT_imports.loc[province].to_numpy()),
                                self.INT_imports.loc[province].index, self.INT_imports.loc[province].index)
            dff = norm_emissions_INT_imports.dot(diag)
            self.F_INT_imports = pd.concat([self.F_INT_imports, dff.T.set_index(
                pd.MultiIndex.from_product([[province], dff.columns.tolist()])).T], axis=1)

    def gimme_symmetric_iot(self):
        """
        Transforms Supply and Use tables to symmetric IO tables and transforms Y from product to industries if
        selected classification is "industry"
        :return: self.A, self.R and self.Y
        """
        self.inv_q = pd.DataFrame(np.diag((1 / self.q.sum(axis=1)).replace(np.inf, 0)), self.q.index, self.q.index)
        self.inv_g = pd.DataFrame(np.diag((1 / self.g.sum()).replace(np.inf, 0)), self.g.columns, self.g.columns)

        if self.assumption == "industry technology" and self.classification == "product":
            self.A = self.U.dot(self.inv_g.dot(self.V.T)).dot(self.inv_q)
            self.R = self.W.dot(self.inv_g.dot(self.V.T)).dot(self.inv_q)
        elif self.assumption == "fixed industry sales structure" and self.classification == "industry":
            self.A = self.V.T.dot(self.inv_q).dot(self.U).dot(self.inv_g)
            self.R = self.W.dot(self.inv_g)
            # TODO check the Y in industries transformation
            self.Y = self.V.dot(self.inv_g).T.dot(self.Y)

    def balance_value_added(self):
        """
        Making sure A+R sums up to 1. Balancing on value added because no impacts on the environment is associated to
        value added.
        :return: updated self.R
        """

        new_value_added_value = 1 - self.A.sum()

        updated_R = (self.R / self.R.sum()) * new_value_added_value
        updated_R = updated_R.fillna(0)

        # checking that is the balance value is different than 1, it's because it's empty industries
        balance = self.A.sum() + updated_R.sum()
        assert balance.loc[~np.isclose(balance, 1)].sum() == 0

        self.R = updated_R

    def extract_environmental_data(self):
        """
        Extracts the data from the NPRI file
        :return: self.F but linked to NAICS codes
        """
        # Tab name changes with selected year, so identify it using "INRP-NPRI"
        emissions = self.NPRI[[i for i in self.NPRI.keys() if "INRP-NPRI" in i][0]]
        emissions.columns = list(zip(emissions.iloc[0].ffill().tolist(), emissions.iloc[2]))
        emissions = emissions.iloc[3:]
        # selecting the relevant columns from the file
        emissions = emissions.loc[:, [i for i in emissions.columns if
                                      (i[1] in
                                       ['NAICS 6 Code', 'CAS Number', 'Substance Name (English)', 'Units', 'Province']
                                       or 'Total' in i[1] and 'Air' in i[0]
                                       or 'Total' in i[1] and 'Water' in i[0]
                                       or 'Total' in i[1] and 'Land' in i[0])]].fillna(0)
        # renaming the columns
        emissions.columns = ['Province', 'NAICS 6 Code', 'CAS Number', 'Substance Name', 'Units', 'Emissions to air',
                             'Emissions to water', 'Emissions to land']

        # somehow the NPRI manages to have entries without NAICS codes... Remove them
        no_naics_code_entries = emissions.loc[:, 'NAICS 6 Code'][emissions.loc[:, 'NAICS 6 Code'] == 0].index
        emissions.drop(no_naics_code_entries, inplace=True)

        # NAICS codes as strings and not integers
        emissions.loc[:, 'NAICS 6 Code'] = emissions.loc[:, 'NAICS 6 Code'].astype('str')

        # extracting metadata for substances
        temp_df = emissions.copy()
        temp_df.set_index('Substance Name', inplace=True)
        temp_df = temp_df.groupby(temp_df.index).head(n=1)
        # separating the metadata for emissions (CAS and units)
        self.emission_metadata = pd.DataFrame('', index=temp_df.index, columns=['CAS Number', 'Unit'])
        for emission in temp_df.index:
            self.emission_metadata.loc[emission, 'CAS Number'] = temp_df.loc[emission, 'CAS Number']
            self.emission_metadata.loc[emission, 'Unit'] = temp_df.loc[emission, 'Units']
        del temp_df

        self.F = pd.pivot_table(data=emissions, index=['Province', 'Substance Name'],
                                columns=['Province', 'NAICS 6 Code'], aggfunc=np.sum).fillna(0)
        # renaming compartments
        self.F.columns.set_levels(['Air', 'Water', 'Soil'], level=0, inplace=True)
        # renaming the names of the columns indexes
        self.F.columns = self.F.columns.rename(['compartment', 'Province', 'NAICS'])
        # reorder multi index to have province as first level
        self.F = self.F.reorder_levels(['Province', 'compartment', 'NAICS'], axis=1)
        # match compartments with emissions and not to provinces
        self.F = self.F.T.unstack('compartment').T[self.F.T.unstack('compartment').T != 0].fillna(0)
        # identify emissions that are in tonnes
        emissions_to_rescale = [i for i in self.emission_metadata.index if
                                self.emission_metadata.loc[i, 'Unit'] == 'tonnes']
        # convert them to kg
        self.F.loc(axis=0)[:, emissions_to_rescale] *= 1000
        self.emission_metadata.loc[emissions_to_rescale, 'Unit'] = 'kg'
        # same thing for emissions in grams
        emissions_to_rescale = [i for i in self.emission_metadata.index if
                                self.emission_metadata.loc[i, 'Unit'] == 'grams']
        self.F.loc(axis=0)[:, emissions_to_rescale] /= 1000
        self.emission_metadata.loc[emissions_to_rescale, 'Unit'] = 'kg'

        # harmonizing emissions across provinces, set to zero if missing initially
        new_index = pd.MultiIndex.from_product(
            [self.matching_dict, self.emission_metadata.sort_index().index, ['Air', 'Water', 'Soil']])
        self.F = self.F.reindex(new_index).fillna(0)

        # harmonizing NAICS codes across provinces this time
        self.F = self.F.T.reindex(
            pd.MultiIndex.from_product([self.F.columns.levels[0], self.F.columns.levels[1]])).T.fillna(0)

    def match_npri_data_to_iots(self):

        total_emissions_origin = self.F.sum().sum()

        # load and format concordances file
        concordance = pd.read_excel(pkg_resources.resource_stream(__name__, '/Data/NPRI_concordance.xlsx'),
                                    self.level_of_detail)
        concordance.set_index('NAICS 6 Code', inplace=True)
        concordance.drop('NAICS 6 Sector Name (English)', axis=1, inplace=True)

        # splitting emissions between private and public sectors
        if self.level_of_detail == 'Summary level':
            self.split_private_public_sectors(NAICS_code=['611210', '611310', '611510'], IOIC_code='GS610')
        elif self.level_of_detail == 'Link-1961 level':
            self.split_private_public_sectors(NAICS_code=['611210', '611510'], IOIC_code='GS611B0')
            self.split_private_public_sectors(NAICS_code='611310', IOIC_code='GS61130')
        elif self.level_of_detail in ['Link-1997 level', 'Detail level']:
            self.split_private_public_sectors(NAICS_code='611210', IOIC_code='GS611200')
            self.split_private_public_sectors(NAICS_code='611310', IOIC_code='GS611300')
            self.split_private_public_sectors(NAICS_code='611510', IOIC_code='GS611A00')

        # switch NAICS codes in self.F for corresponding IOIC codes (from concordances file)
        IOIC_index = []
        for NAICS in self.F.columns:
            try:
                IOIC_index.append((NAICS[0], concordance.loc[int(NAICS[1]), 'IOIC']))
            except ValueError:
                IOIC_index.append(NAICS)
        self.F.columns = pd.MultiIndex.from_tuples(IOIC_index)

        # adding emissions from same sectors together (summary level is more aggregated than NAICS 6 Code)
        self.F = self.F.groupby(self.F.columns, axis=1).sum()
        # reordering columns
        self.F = self.F.T.reindex(
            pd.MultiIndex.from_product([self.matching_dict, [i[0] for i in self.industries]])).T.fillna(0)
        # changing codes for actual names of the sectors
        self.F.columns = pd.MultiIndex.from_product([self.matching_dict, [i[1] for i in self.industries]])

        # assert that nearly all emissions present in the NPRI were successfully transferred in self.F
        assert self.F.sum().sum() / total_emissions_origin > 0.98
        assert self.F.sum().sum() / total_emissions_origin < 1.02

    def match_ghg_accounts_to_iots(self):
        """
        Method matching GHG accounts to IOIC classification selected by the user
        :return: self.F and self.FY with GHG flows included
        """
        GHG = pd.read_excel(pkg_resources.resource_stream(__name__, '/Data/GHG_emissions_by_gas_RY2017-RY2018.xlsx'),
                          'L61 ghg emissions by gas')

        if self.year in GHG.loc[:, 'Reference Year'].values:
            GHG = GHG.loc[
                [i for i in GHG.index if GHG.loc[i, 'Reference Year'] == self.year and GHG.Geography[i] != 'Canada']]
        else:
            GHG = GHG.loc[
                [i for i in GHG.index if GHG.loc[i, 'Reference Year'] == 2017 and GHG.Geography[i] != 'Canada']]
        # kilotonnes to kgs
        GHG.loc[:, ['CO2', 'CH4', 'N2O']] *= 1000000

        # start with the households emissions
        Household_GHG = GHG.loc[[i for i in GHG.index if 'PEH' in GHG.loc[i, 'IOIC']]]
        Household_GHG = Household_GHG.groupby('Geography').sum().drop('Reference Year', axis=1).T
        Household_GHG.index = ['Carbon dioxide', 'Methane', 'Dinitrogen monoxide']
        Household_GHG.columns = [{v: k for k, v in self.matching_dict.items()}[i] for i in Household_GHG.columns]
        Household_GHG.columns = pd.MultiIndex.from_product(
            [Household_GHG.columns, ['Household final consumption expenditure']])
        self.FY = pd.DataFrame(0, Household_GHG.index, self.Y.columns).merge(Household_GHG, 'right').fillna(0)
        self.FY.index = pd.MultiIndex.from_product([Household_GHG.index.tolist(), ['Air']])
        # spatialization
        self.FY = pd.concat([self.FY] * len(self.FY.columns.levels[0]), axis=0)
        self.FY.index = pd.MultiIndex.from_product(
            [list(self.matching_dict.keys()), ['Carbon dioxide', 'Methane', 'Dinitrogen monoxide'], ['Air']])
        for province in self.FY.columns.levels[0]:
            self.FY.loc[[i for i in self.FY.index.levels[0] if i != province], province] = 0

        # Now the emissions from production
        GHG.set_index(pd.MultiIndex.from_tuples(tuple(
            list(zip([{v: k for k, v in self.matching_dict.items()}[i] for i in GHG.Geography], GHG.IOIC.tolist())))),
                      inplace=True)
        GHG.drop(['IOIC', 'Reference Year', 'Geography', 'Description', 'F_Description'], axis=1, inplace=True)
        GHG.drop([i for i in GHG.index if re.search(r'^FC', i[1])
                  or re.search(r'^PEH', i[1])
                  or re.search(r'^Total', i[1])], inplace=True)
        GHG.columns = ['Carbon dioxide', 'Methane', 'Dinitrogen monoxide']

        concordance = pd.read_excel(pkg_resources.resource_stream(__name__, '/Data/GHG_concordance.xlsx'),
                                    self.level_of_detail)
        concordance.set_index('GHG codes', inplace=True)

        if self.level_of_detail in ['Summary level', 'Link-1961 level']:
            # transform GHG accounts sectors to IOIC sectors
            GHG.index = pd.MultiIndex.from_tuples([(i[0], concordance.loc[i[1], 'IOIC']) for i in GHG.index])
            # some sectors are not linked to IOIC (specifically weird Canabis sectors), drop them
            if len([i for i in GHG.index if type(i[1]) == float]) != 0:
                GHG.drop([i for i in GHG.index if type(i[1]) == float], inplace=True)
            # grouping emissions from same sectors
            GHG = GHG.groupby(GHG.index).sum()
            GHG.index = pd.MultiIndex.from_tuples(GHG.index)
            # reindex to make sure dataframe is ordered as in dictionary
            GHG = GHG.reindex(pd.MultiIndex.from_product([self.matching_dict, [i[0] for i in self.industries]]))
            # switching codes for readable names
            GHG.index = pd.MultiIndex.from_product([self.matching_dict, [i[1] for i in self.industries]])

            # spatializing GHG emissions in case we later regionalize impacts (even though it's useless for climate change)
            GHG = pd.concat([GHG] * len(GHG.index.levels[0]), axis=1)
            GHG.columns = pd.MultiIndex.from_product(
                [self.matching_dict, ['Carbon dioxide', 'Methane', 'Dinitrogen monoxide'], ['Air']])
            # emissions takes place in the province of the trade
            for province in GHG.index.levels[0]:
                GHG.loc[province, [i for i in GHG.index.levels[0] if i != province]] = 0
            # add GHG emissions to other pollutants
            self.F = pd.concat([self.F, GHG.T])
            self.F.index = pd.MultiIndex.from_tuples(self.F.index)

        elif self.level_of_detail in ['Link-1997 level', 'Detail level']:
            # dropping empty sectors (mostly Cannabis related)
            to_drop = concordance.loc[concordance.loc[:, 'IOIC'].isna()].index
            concordance.drop(to_drop, inplace=True)
            ghgs = pd.DataFrame()
            for code in concordance.index:
                # L97 and D levels are more precise than GHG accounts, we use market share to distribute GHGs
                sectors_to_split = [i[1] for i in self.industries if
                                    i[0] in concordance.loc[code].dropna().values.tolist()]
                output_sectors_to_split = self.V.loc[:,
                                          [i for i in self.V.columns if i[1] in sectors_to_split]].sum()
                share_sectors_to_split = pd.DataFrame(0, output_sectors_to_split.index,
                                                      ['Carbon dioxide', 'Methane', 'Dinitrogen monoxide'])
                for province in output_sectors_to_split.index.levels[0]:
                    df = ((output_sectors_to_split.loc[province] / output_sectors_to_split.loc[province].sum()).fillna(0).values)
                    # hardcoded 3 because 3 GHGs: CO2, CH4, N2O
                    share_sectors_to_split.loc[province] = (pd.DataFrame([df] * 3,
                                                                         index=['Carbon dioxide', 'Methane', 'Dinitrogen monoxide'],
                                                                         columns=sectors_to_split).T
                                                            * GHG.loc(axis=0)[:, code].loc[province].values).values
                ghgs = pd.concat([ghgs, share_sectors_to_split])
            # spatializing GHG emissions
            ghgs = pd.concat([ghgs] * len(ghgs.index.levels[0]), axis=1)
            ghgs.columns = pd.MultiIndex.from_product(
                [list(self.matching_dict.keys()), ['Carbon dioxide', 'Methane', 'Dinitrogen monoxide'], ['Air']])
            for province in ghgs.columns.levels[0]:
                ghgs.loc[[i for i in ghgs.index.levels[0] if i != province], province] = 0

            # adding GHG accounts to pollutants
            self.F = pd.concat([self.F, ghgs.T])

            # reindexing
            self.F = self.F.reindex(self.U.columns, axis=1)

        self.emission_metadata.loc['Carbon dioxide', 'CAS Number'] = '000124-38-9'
        self.emission_metadata.loc['Carbon dioxide', 'Unit'] = 'kg'
        self.emission_metadata.loc['Methane', 'CAS Number'] = '000074-82-8'
        self.emission_metadata.loc['Methane', 'Unit'] = 'kg'
        self.emission_metadata.loc['Dinitrogen monoxide', 'CAS Number'] = '010024-97-2'
        self.emission_metadata.loc['Dinitrogen monoxide', 'Unit'] = 'kg'

    def match_water_accounts_to_iots(self):
        """
        Method matching water accounts to IOIC classification selected by the user
        :return: self.F and self.FY with GHG flows included
        """
        # load the water use data from STATCAN
        water = pd.read_csv(pkg_resources.resource_stream(__name__, '/Data/water_use.csv'))

        # Only odd years from 2009 to 2017
        if self.year == 2010:
            year_for_water = 2011
        elif self.year == 2012:
            year_for_water = 2013
        elif self.year == 2014:
            year_for_water = 2015
        elif self.year == 2016:
            year_for_water = 2015
        elif self.year > 2017:
            year_for_water = 2017
        else:
            year_for_water = self.year
        # select the year of the data
        water = water.loc[
            [i for i in water.index if water.REF_DATE[i] == int(year_for_water)], ['Sector', 'VALUE']].fillna(0)

        # convert into cubic meters
        water.VALUE *= 1000

        # water use from households
        FD_water = water.loc[[i for i in water.index if water.Sector[i] == 'Households']]
        # national water use will be distributed depending on the amount of $ spent by households in a given province
        provincial_FD_consumption_distribution = self.Y.loc(axis=1)[:,
                                                 'Household final consumption expenditure'].sum() / self.Y.loc(
            axis=1)[:, 'Household final consumption expenditure'].sum().sum()
        FD_water = provincial_FD_consumption_distribution * FD_water.VALUE.values
        # spatializing
        FD_water = pd.concat([FD_water] * len(FD_water.index.levels[0]), axis=1)
        FD_water.columns = pd.MultiIndex.from_product([self.matching_dict.keys(), ['Water'], ['Water']])
        FD_water = FD_water.T
        for province in FD_water.index.levels[0]:
            FD_water.loc[province, [i for i in FD_water.columns if i[0] != province]] = 0
        FD_water = FD_water.T.reindex(self.Y.columns).T.fillna(0)
        self.FY = pd.concat([self.FY, FD_water])

        # format the names of the sector to match those used up till then
        water = water.loc[[i for i in water.index if '[' in water.Sector[i]]]
        water.Sector = [i.split('[')[1].split(']')[0] for i in water.Sector]
        water.drop([i for i in water.index if re.search(r'^FC', water.Sector.loc[i])], inplace=True)
        water.set_index('Sector', inplace=True)

        # load concordances matching water use data classification to the different classifications used in OpenIO
        concordance = pd.read_excel(pkg_resources.resource_stream(__name__, '/Data/water_concordance.xlsx'),
                                    self.level_of_detail)
        concordance.set_index('Sector', inplace=True)
        # dropping potential empty sectors (mostly Cannabis related)
        to_drop = concordance.loc[concordance.loc[:, 'IOIC'].isna()].index
        concordance.drop(to_drop, inplace=True)

        water_flows = pd.DataFrame()
        if self.level_of_detail in ['Link-1961 level', 'Link-1997 level', 'Detail level']:
            for code in concordance.index:
                # Detail level is more precise than water accounts, we use market share to distribute water flows
                sectors_to_split = [i[1] for i in self.industries if
                                    i[0] in concordance.loc[code].dropna().values.tolist()]
                output_sectors_to_split = self.V.loc[:,
                                          [i for i in self.V.columns if i[1] in sectors_to_split]].sum()

                share_sectors_to_split = output_sectors_to_split / output_sectors_to_split.sum() * water.loc[
                    code, 'VALUE']
                water_flows = pd.concat([water_flows, share_sectors_to_split])
        elif self.level_of_detail == 'Summary level':
            water = pd.concat([water, concordance], axis=1)
            water.set_index('IOIC', inplace=True)
            water = water.groupby(water.index).sum()
            water.index = [dict(self.industries)[i] for i in water.index]
            water = water.reindex([i[1] for i in self.industries]).fillna(0)
            water_flows = pd.DataFrame()
            for sector in water.index:
                water_split = self.g.sum().loc(axis=0)[:, sector] / self.g.sum().loc(axis=0)[:, sector].sum() * \
                              water.loc[sector, 'VALUE']
                water_flows = pd.concat([water_flows, water_split])

        water_flows = water_flows.groupby(water_flows.index).sum().fillna(0)
        # multi index for the win
        water_flows.index = pd.MultiIndex.from_tuples(water_flows.index)
        water_flows.columns = ['Water']

        # spatializing water flows
        water_flows = pd.concat([water_flows.T] * len(water_flows.index.levels[0]))
        water_flows.index = pd.MultiIndex.from_product([self.matching_dict.keys(), ['Water'], ['Water']])
        water_flows = water_flows.T.reindex(self.F.columns).T
        for province in water_flows.index.levels[0]:
            water_flows.loc[province, [i for i in water_flows.columns if i[0] != province]] = 0

        # fillna(0) for cannabis industries
        self.F = pd.concat([self.F, water_flows]).fillna(0)

    def characterization_matrix(self):
        """
        Produces a characterization matrix from IMPACT World+ file
        :return: self.C, self.methods_metadata
        """

        IW = pd.read_excel(
            pkg_resources.resource_stream(
                __name__, '/Data/IW+ 1_48 EP_1_30 MP_as DB_rules_compliance_with_manually_added_CF.xlsx'))

        pivoting = pd.pivot_table(IW, values='CF value', index=('Impact category', 'CF unit'),
                                  columns=['Elem flow name', 'Compartment', 'Sub-compartment']).fillna(0)

        concordance = pd.read_excel(pkg_resources.resource_stream(__name__, '/Data/NPRI_IW_concordance.xlsx'))
        concordance.set_index('NPRI flows', inplace=True)

        # adding GHGs to the list of pollutants
        concordance = pd.concat([concordance, pd.DataFrame(['Carbon dioxide, fossil', 'Methane, fossil', 'Dinitrogen monoxide'],
                                                           index=['Carbon dioxide', 'Methane', 'Dinitrogen monoxide'],
                                                           columns=['IMPACT World+ flows'])])

        self.C = pd.DataFrame(0, pivoting.index, self.F.index)
        for flow in self.C.columns:
            try:
                if concordance.loc[flow[1], 'IMPACT World+ flows'] is not None:
                    self.C.loc[:, flow] = pivoting.loc[:,
                                          [(concordance.loc[flow[1], 'IMPACT World+ flows'], flow[2],
                                            '(unspecified)')]].values
            except KeyError:
                pass

        self.C.loc[('Water use', 'm3'), [i for i in self.C.columns if i[1] == 'Water']] = 1

        # some methods of IMPACT World+ do not make sense in our context, remove them
        self.C.drop(['Fossil and nuclear energy use',
                     'Ionizing radiations',
                     'Ionizing radiation, ecosystem quality',
                     'Ionizing radiation, human health',
                     'Land occupation, biodiversity',
                     'Land transformation, biodiversity',
                     'Mineral resources use',
                     'Thermally polluted water',
                     'Water availability, freshwater ecosystem',
                     'Water availability, human health',
                     'Water availability, terrestrial ecosystem',
                     'Water scarcity'], axis=0, level=0, inplace=True)

        # importing characterization matrix IMPACT World+/exiobase
        self.C_INT_imports = pd.read_csv(pkg_resources.resource_stream(__name__, '/Data/C_exio_IW_1.30_1.48.csv'))
        self.C_INT_imports.set_index('Unnamed: 0', inplace=True)
        self.C_INT_imports.index.name = None

        self.C_INT_imports.index = pd.MultiIndex.from_tuples(list(zip(
            [i.split(' (')[0] for i in self.C_INT_imports.index],
            [i.split(' (')[1].split(')')[0] for i in self.C_INT_imports.index])))

        self.C_INT_imports.drop(['Fossil and nuclear energy use',
                                 'Ionizing radiations',
                                 'Ionizing radiation, ecosystem quality',
                                 'Ionizing radiation, human health',
                                 'Land occupation, biodiversity',
                                 'Land transformation, biodiversity',
                                 'Mineral resources use',
                                 'Thermally polluted water',
                                 'Water availability, freshwater ecosystem',
                                 'Water availability, human health',
                                 'Water availability, terrestrial ecosystem',
                                 'Water scarcity'], axis=0, level=0, inplace=True)
        # adding water use to exiobase flows to match with water use from STATCAN physical accounts
        # water use in exiobase is identified through "water withdrawal" and NOT "water consumption"
        adding_water_use = pd.DataFrame(0, index=pd.MultiIndex.from_product([['Water use'], ['m3']]),
                                        columns=self.F_INT_imports.index)
        # STATCAN excluded water use due to hydroelectricity from their accounts, we keep consistency by removing them too
        adding_water_use.loc[:, [i for i in self.F_INT_imports.index if 'Water Withdrawal' in i and (
                'hydro' not in i or 'tide' not in i)]] = 1
        self.C_INT_imports = pd.concat([self.C_INT_imports, adding_water_use])
        # forcing the match with self.C (annoying parentheses for climate change long and short term)
        self.C_INT_imports.index = self.C.index

        self.methods_metadata = pd.DataFrame(self.C.index.tolist(), columns=['Impact category', 'unit'])
        self.methods_metadata = self.methods_metadata.set_index('Impact category')

        self.balance_flows(concordance)

    def balance_flows(self, concordance):
        """
        Some flows from the NPRI trigger some double counting if left unattended. This method deals with these flows
        :return: balanced self.F
        """

        # VOCs
        rest_of_voc = [i for i in concordance.index if 'Speciated VOC' in i and concordance.loc[i].isna().iloc[0]]
        df = self.F.loc(axis=0)[:, rest_of_voc]
        self.F.loc(axis=0)[:, 'Volatile organic compounds', 'Air'] += df.groupby(level=0).sum().values
        self.F.drop(self.F.loc(axis=0)[:, rest_of_voc].index, inplace=True)

        # PMs, only take highest value flow as suggested by the NPRI team:
        # [https://www.canada.ca/en/environment-climate-change/services/national-pollutant-release-inventory/using-interpreting-data.html]
        for sector in self.F.columns:
            little_pm = self.F.loc[(sector[0], 'PM2.5', 'Air'), sector]
            big_pm = self.F.loc[(sector[0], 'PM10', 'Air'), sector]
            unknown_size = self.F.loc[(sector[0], 'Total particulate matter', 'Air'), sector]
            if little_pm >= big_pm:
                if little_pm >= unknown_size:
                    self.F.loc[(sector[0], 'PM10', 'Air'), sector] = 0
                    self.F.loc[(sector[0], 'Total particulate matter', 'Air'), sector] = 0
                else:
                    self.F.loc[(sector[0], 'PM10', 'Air'), sector] = 0
                    self.F.loc[(sector[0], 'PM2.5', 'Air'), sector] = 0
            else:
                if big_pm > unknown_size:
                    self.F.loc[(sector[0], 'PM2.5', 'Air'), sector] = 0
                    self.F.loc[(sector[0], 'Total particulate matter', 'Air'), sector] = 0
                else:
                    self.F.loc[(sector[0], 'PM10', 'Air'), sector] = 0
                    self.F.loc[(sector[0], 'PM2.5', 'Air'), sector] = 0

        # we modified flows in self.F, modify self.C accordingly
        self.C = self.C.loc[:, self.F.index].fillna(0)

    def normalize_flows(self):
        """
        Produce normalized environmental extensions
        :return: self.S and self.F with product classification if it's been selected
        """

        self.F = self.F.sort_index()

        if self.classification == 'industry':
            self.S = self.F.dot(self.inv_g)

        if self.classification == 'product':
            self.F = self.F.dot(self.V.dot(self.inv_g).T)
            self.S = self.F.dot(self.inv_q)
            if self.exiobase_folder:
                self.S_INT_imports = self.F_INT_imports.dot(self.inv_q)

        # adding empty flows to FY to allow multiplication with self.C
        self.FY = pd.concat([pd.DataFrame(0, self.F.index, self.Y.columns), self.FY])
        self.FY = self.FY.groupby(self.FY.index).sum()

    def calc(self):
        """
        Method to calculate the Leontief inverse and get total impacts
        :return: self.L (total requirements), self.E (total emissions), self.D (total impacts)
        """
        I = pd.DataFrame(np.eye(len(self.A)), self.A.index, self.A.columns)
        self.L = pd.DataFrame(np.linalg.solve(I - self.A, I), self.A.index, I.columns)
        self.E = self.S.dot(self.L).dot(self.Y) + self.FY
        if self.exiobase_folder:
            self.E_INT_imports = self.S_INT_imports.dot(self.L).dot(self.Y)
            self.D = self.C.dot(self.E) + self.C_INT_imports.dot(self.E_INT_imports)
        else:
            self.D = self.C.dot(self.E)

    def split_private_public_sectors(self, NAICS_code, IOIC_code):
        """
        Support method to split equally emissions from private and public sectors
        :param NAICS_code: [string or list] the NAICS code(s) whose emissions will be split
        :param IOIC_code: [string] the IOIC_code inhereting the split emissions (will be private or public sector)
        :return: updated self.F
        """
        df = self.F.loc(axis=1)[:, NAICS_code].copy()
        if type(NAICS_code) == list:
            df.columns = pd.MultiIndex.from_product([self.matching_dict, [IOIC_code] * len(NAICS_code)])
        elif type(NAICS_code) == str:
            df.columns = pd.MultiIndex.from_product([self.matching_dict, [IOIC_code]])
        self.F = pd.concat([self.F, df / 2], axis=1)
        self.F.loc(axis=1)[:, NAICS_code] /= 2

    def produce_npri_iw_concordance_file(self):
        """
        Method to obtain the NPRI_IW_concordance.xlsx file (for reproducibility)
        :return: the NPRI_IW_concordance.xlsx file
        """

        IW = pd.read_excel(
            pkg_resources.resource_stream(
                __name__, '/Data/IW+ 1_48 EP_1_30 MP_as DB_rules_compliance_with_manually_added_CF.xlsx'))

        df = IW.set_index('CAS number')
        df.index = [str(i) for i in df.index]
        df = df.groupby(df.index).head(n=1)

        concordance_IW = dict.fromkeys(self.F.index.levels[0])

        # match pollutants using CAS numbers
        for pollutant in concordance_IW:
            match_CAS = ''
            try:
                if len(self.emission_metadata.loc[(pollutant, 'Air'), 'CAS Number'].split('-')[0]) == 2:
                    match_CAS = '0000' + self.emission_metadata.loc[(pollutant, 'Air'), 'CAS Number']
                elif len(self.emission_metadata.loc[(pollutant, 'Air'), 'CAS Number'].split('-')[0]) == 3:
                    match_CAS = '000' + self.emission_metadata.loc[(pollutant, 'Air'), 'CAS Number']
                elif len(self.emission_metadata.loc[(pollutant, 'Air'), 'CAS Number'].split('-')[0]) == 4:
                    match_CAS = '00' + self.emission_metadata.loc[(pollutant, 'Air'), 'CAS Number']
                elif len(self.emission_metadata.loc[(pollutant, 'Air'), 'CAS Number'].split('-')[0]) == 5:
                    match_CAS = '0' + self.emission_metadata.loc[(pollutant, 'Air'), 'CAS Number']
                elif len(self.emission_metadata.loc[(pollutant, 'Air'), 'CAS Number'].split('-')[0]) == 6:
                    match_CAS = self.emission_metadata.loc[(pollutant, 'Air'), 'CAS Number']
                try:
                    concordance_IW[pollutant] = [df.loc[i, 'Elem flow name'] for i in df.index if i == match_CAS][0]

                except IndexError:
                    pass
            except KeyError:
                pass

        # hardcoding what could not be matched using CAS number
        concordance_IW['Ammonia (total)'] = 'Ammonia'
        concordance_IW['Fluorine'] = 'Fluorine'
        concordance_IW['PM10 - Particulate matter <=10 microns'] = 'Particulates, < 10 um'
        concordance_IW['PM2.5 - Particulate matter <=2.5 microns'] = 'Particulates, < 2.5 um'
        concordance_IW['Total particulate matter'] = 'Particulates, unspecified'
        concordance_IW['Speciated VOC - Cycloheptane'] = 'Cycloheptane'
        concordance_IW['Speciated VOC - Cyclohexene'] = 'Cyclohexene'
        concordance_IW['Speciated VOC - Cyclooctane'] = 'Cyclooctane'
        concordance_IW['Speciated VOC - Hexane'] = 'Hexane'
        concordance_IW[
            'Volatile organic compounds'] = 'NMVOC, non-methane volatile organic compounds, unspecified origin'

        # proxies, NOT A 1 FOR 1 MATCH but better than no characterization factor
        concordance_IW['HCFC-123 (all isomers)'] = 'Ethane, 2-chloro-1,1,1,2-tetrafluoro-, HCFC-123'
        concordance_IW['HCFC-124 (all isomers)'] = 'Ethane, 2,2-dichloro-1,1,1-trifluoro-, HCFC-124'
        concordance_IW['Nonylphenol and its ethoxylates'] = 'Nonylphenol'
        concordance_IW['Phosphorus (yellow or white only)'] = 'Phosphorus'
        concordance_IW['Phosphorus (total)'] = 'Phosphorus'
        concordance_IW['PAHs, total unspeciated'] = 'Hydrocarbons, aromatic'
        concordance_IW['Aluminum oxide (fibrous forms only)'] = 'Aluminium'
        concordance_IW['Antimony (and its compounds)'] = 'Antimony'
        concordance_IW['Arsenic (and its compounds)'] = 'Arsenic'
        concordance_IW['Cadmium (and its compounds)'] = 'Cadmium'
        concordance_IW['Chromium (and its compounds)'] = 'Chromium'
        concordance_IW['Hexavalent chromium (and its compounds)'] = 'Chromium VI'
        concordance_IW['Cobalt (and its compounds)'] = 'Cobalt'
        concordance_IW['Copper (and its compounds)'] = 'Copper'
        concordance_IW['Lead (and its compounds)'] = 'Lead'
        concordance_IW['Nickel (and its compounds)'] = 'Nickel'
        concordance_IW['Mercury (and its compounds)'] = 'Mercury'
        concordance_IW['Manganese (and its compounds)'] = 'Manganese'
        concordance_IW['Selenium (and its compounds)'] = 'Selenium'
        concordance_IW['Silver (and its compounds)'] = 'Silver'
        concordance_IW['Thallium (and its compounds)'] = 'Thallium'
        concordance_IW['Zinc (and its compounds)'] = 'Zinc'
        concordance_IW['Speciated VOC - Butane  (all isomers)'] = 'Butane'
        concordance_IW['Speciated VOC - Butene  (all isomers)'] = '1-Butene'
        concordance_IW['Speciated VOC - Anthraquinone (all isomers)'] = 'Anthraquinone'
        concordance_IW['Speciated VOC - Decane  (all isomers)'] = 'Decane'
        concordance_IW['Speciated VOC - Dodecane  (all isomers)'] = 'Dodecane'
        concordance_IW['Speciated VOC - Heptane  (all isomers)'] = 'Heptane'
        concordance_IW['Speciated VOC - Nonane  (all isomers)'] = 'Nonane'
        concordance_IW['Speciated VOC - Octane  (all isomers)'] = 'N-octane'
        concordance_IW['Speciated VOC - Pentane (all isomers)'] = 'Pentane'
        concordance_IW['Speciated VOC - Pentene (all isomers)'] = '1-Pentene'

        return pd.DataFrame.from_dict(concordance_IW, orient='index')

    def export(self, filepath='', format=''):
        """
        Function to export in the chosen format.
        :param filepath: the path where to store the export file
        :param format: available formats 'csv', 'excel', 'pickle', 'json'
        :return: nothing
        """

        if not filepath:
            print("Please provide a filepath")
            return
        if not format:
            print("Please enter a format")
            return

        N = self.C.dot(self.S).dot(self.L)
        B = self.S.dot(self.L)

        def flat_multiindex(df):
            df.index = df.index.tolist()
            df.columns = df.columns.tolist()
        flat_multiindex(self.A)
        flat_multiindex(self.Y)
        flat_multiindex(self.R)
        flat_multiindex(self.S)
        flat_multiindex(self.FY)
        flat_multiindex(self.C)
        flat_multiindex(self.L)
        flat_multiindex(self.E)
        flat_multiindex(self.D)
        flat_multiindex(B)
        flat_multiindex(N)

        def remove_zeros(df):
            return df.replace({0: np.nan})
        self.A = remove_zeros(self.A)
        self.Y = remove_zeros(self.Y)
        self.R = remove_zeros(self.R)
        self.S = remove_zeros(self.S)
        self.FY = remove_zeros(self.FY)
        self.C = remove_zeros(self.C)
        self.L = remove_zeros(self.L)
        self.E = remove_zeros(self.E)
        self.D = remove_zeros(self.D)
        B = remove_zeros(B)
        N = remove_zeros(N)

        if format == 'excel':
            writer = pd.ExcelWriter(filepath, engine='xlsxwriter')

            self.A.to_excel(writer, 'A')
            self.Y.to_excel(writer, 'Y')
            self.R.to_excel(writer, 'R')
            self.S.to_excel(writer, 'S')
            self.FY.to_excel(writer, 'FY')
            self.C.to_excel(writer, 'C')
            self.L.to_excel(writer, 'L')
            self.E.to_excel(writer, 'E')
            self.D.to_excel(writer, 'D')
            B.to_excel(writer, 'B')
            N.to_excel(writer, 'N')

            writer.save()

        else:
            print('Format requested not implemented yet.')

# ------------------------------------------------ DEPRECATED ---------------------------------------------------------
    def old_province_import_export(self, province_trade_file):
        """
        Method extracting and formatting inter province imports/exports
        :return: modified self.U, self.V, self.W, self.Y
        """

        province_trade_file = province_trade_file

        province_trade_file.Origin = [{v: k for k, v in self.matching_dict.items()}[i.split(') ')[1]] if (
                    ')' in i and i != '(81) Canadian territorial enclaves abroad') else i for i in
                                    province_trade_file.Origin]
        province_trade_file.Destination = [{v: k for k, v in self.matching_dict.items()}[i.split(') ')[1]] if (
                    ')' in i and i != '(81) Canadian territorial enclaves abroad') else i for i in
                                         province_trade_file.Destination]
        # extracting and formatting supply for each province
        province_trade = pd.pivot_table(data=province_trade_file, index='Destination', columns=['Origin', 'Product'])

        province_trade = province_trade.loc[
            [i for i in province_trade.index if i in self.matching_dict], [i for i in province_trade.columns if
                                                                                i[1] in self.matching_dict]]
        province_trade *= 1000
        province_trade.columns = [(i[1], i[2].split(': ')[1]) if ':' in i[2] else i for i in
                                     province_trade.columns]
        province_trade.drop([i for i in province_trade.columns if i[1] not in [i[1] for i in self.commodities]],
                            axis=1, inplace=True)
        province_trade.columns = pd.MultiIndex.from_tuples(province_trade.columns)
        for province in province_trade.index:
            province_trade.loc[province, province] = 0

        for importing_province in province_trade.index:
            U_Y = pd.concat([self.U.loc[importing_province, importing_province],
                             self.Y.loc[importing_province, importing_province]], axis=1)
            total_imports = province_trade.groupby(level=1,axis=1).sum().loc[importing_province]
            index_commodity = [i[1] for i in self.commodities]
            total_imports = total_imports.reindex(index_commodity).fillna(0)
            initial_distribution = ((U_Y.T / (U_Y.sum(axis=1))) * total_imports).T.fillna(0)

            # Remove changes in inventories as imports will not go directly into this category
            initial_distribution.drop(["Changes in inventories"], axis=1, inplace=True)
            U_Y.drop(["Changes in inventories"], axis=1, inplace=True)
            # imports cannot be allocated to negative gross fixed capital formation as it is probably not importing if
            # it's transferring ownership for a given product
            initial_distribution.loc[initial_distribution.loc[:, 'Gross fixed capital formation'] < 0,
                                     'Gross fixed capital formation'] = 0
            U_Y.loc[U_Y.loc[:, 'Gross fixed capital formation'] < 0, 'Gross fixed capital formation'] = 0

            # Remove products where total imports exceed consumption, or there are actually no imports
            bad_ix_excess_imports = total_imports[(U_Y.sum(1) - total_imports) < 0].index.to_list()
            bad_ix_no_import = total_imports[total_imports <= 0].index.to_list()
            bad_ix = bad_ix_excess_imports + bad_ix_no_import
            initial_distribution = initial_distribution.drop(bad_ix, axis=0)
            U_Y = U_Y.drop(bad_ix, axis=0)
            total_imports = total_imports.drop(bad_ix)

            # pyomo optimization (see code at the end)
            Ui, S_imports, S_positive = reconcile_entire_region(U_Y, initial_distribution, total_imports)

            # add index entries that are null
            Ui = Ui.reindex([i[1] for i in self.commodities]).fillna(0)

            # remove really small values (< 1$) coming from optimization
            Ui = Ui[Ui > 1].fillna(0)

            # distribution balance imports to the different exporting regions
            final_demand_imports = [i for i in Ui.columns if i not in self.U.columns.levels[1]]
            for exporting_province in province_trade.index:
                if importing_province != exporting_province:
                    df = ((Ui.T * (province_trade / province_trade.sum()).fillna(0).loc[
                        exporting_province, importing_province]).T).reindex(Ui.index).fillna(0)
                    # assert index and columns are the same before using .values
                    assert all(self.U.loc[exporting_province, importing_province].index == df.loc[:,
                                                                                             self.U.columns.levels[
                                                                                                 1]].reindex(
                        self.U.loc[exporting_province, importing_province].columns, axis=1).index)
                    assert all(self.U.loc[exporting_province, importing_province].columns == df.loc[:,
                                                                                               self.U.columns.levels[
                                                                                                   1]].reindex(
                        self.U.loc[exporting_province, importing_province].columns, axis=1).columns)
                    # assign new values into self.U and self.Y
                    self.U.loc[exporting_province, importing_province] = df.loc[:,
                                                                           self.U.columns.levels[1]].reindex(
                        self.U.loc[exporting_province, importing_province].columns, axis=1).values
                    self.Y.loc[exporting_province, importing_province].update(df.loc[:, final_demand_imports])

            # remove inter-provincial trade from intra-provincial trade
            self.U.loc[importing_province, importing_province].update(
                self.U.loc[importing_province, importing_province] - self.U.loc[
                    [i for i in self.matching_dict if i != importing_province], importing_province].groupby(
                    level=1).sum())
            self.Y.loc[importing_province, importing_province].update(
                self.Y.loc[importing_province, importing_province] - self.Y.loc[
                    [i for i in self.matching_dict if i != importing_province], importing_province].groupby(
                    level=1).sum())

    def old_match_ghg_accounts_to_iots(self):
        """
        Method was for aggregated GHG accounts. New method works with disaggregated accounts.

        Method matching GHG accounts to IOIC classification selected by the user
        :return: self.F and self.FY with GHG flows included
        """
        GHG = pd.read_csv(pkg_resources.resource_stream(__name__, '/Data/GHG_emissions.csv'))
        GHG = GHG.loc[[i for i in GHG.index if GHG.REF_DATE[i] == self.year and GHG.GEO[i] != 'Canada']]
        # kilotonnes to kg CO2e
        GHG.VALUE *= 1000000

        FD_GHG = GHG.loc[[i for i in GHG.index if GHG.Sector[i] == 'Total, households']]
        FD_GHG.GEO = [{v: k for k, v in self.matching_dict.items()}[i] for i in FD_GHG.GEO]
        FD_GHG = FD_GHG.pivot_table(values='VALUE', index=['GEO', 'Sector'])
        FD_GHG.columns = [('', 'GHGs', '')]
        FD_GHG.index.names = (None, None)
        FD_GHG.index = pd.MultiIndex.from_product([self.matching_dict, ['Household final consumption expenditure']])

        GHG = GHG.loc[[i for i in GHG.index if '[' in GHG.Sector[i]]]
        GHG.Sector = [i.split('[')[1].split(']')[0] for i in GHG.Sector]
        GHG.GEO = [{v: k for k, v in self.matching_dict.items()}[i] for i in GHG.GEO]
        GHG = GHG.pivot_table(values='VALUE', index=['GEO', 'Sector'])
        GHG.columns = [('', 'GHGs', '')]
        GHG.index.names = (None, None)
        # reindex to have the same number of sectors covered per province
        GHG = GHG.reindex(pd.MultiIndex.from_product([self.matching_dict, GHG.index.levels[1]])).fillna(0)
        # removing the fictive sectors
        GHG.drop([i for i in GHG.index if re.search(r'^FC', i[1])], inplace=True)

        concordance = pd.read_excel(pkg_resources.resource_stream(__name__, '/Data/GHG_concordance.xlsx'),
                                    self.level_of_detail)
        concordance.set_index('GHG codes', inplace=True)

        if self.level_of_detail in ['Summary level', 'Link-1961 level']:
            # transform GHG accounts sectors to IOIC sectors
            GHG.index = pd.MultiIndex.from_tuples([(i[0], concordance.loc[i[1], 'IOIC']) for i in GHG.index])
            # some sectors are not linked to IOIC (specifically weird Canabis sectors), drop them
            if len([i for i in GHG.index if type(i[1]) == float]) != 0:
                GHG.drop([i for i in GHG.index if type(i[1]) == float], inplace=True)
            # grouping emissions from same sectors
            GHG = GHG.groupby(GHG.index).sum()
            GHG.index = pd.MultiIndex.from_tuples(GHG.index)
            # reindex to make sure dataframe is ordered as in dictionary
            GHG = GHG.reindex(pd.MultiIndex.from_product([self.matching_dict, [i[0] for i in self.industries]]))
            # switching codes for readable names
            GHG.index = pd.MultiIndex.from_product([self.matching_dict, [i[1] for i in self.industries]])

            # spatializing GHG emissions in case we later regionalize impacts (even though it's useless for climate change)
            GHG = pd.concat([GHG] * len(GHG.index.levels[0]), axis=1)
            GHG.columns = pd.MultiIndex.from_product([self.matching_dict, ['GHGs'], ['Air']])
            # emissions takes place in the province of the trade
            for province in GHG.index.levels[0]:
                GHG.loc[province, [i for i in GHG.index.levels[0] if i != province]] = 0
            # add GHG emissions to other pollutants
            self.F = pd.concat([self.F, GHG.T])
            self.F.index = pd.MultiIndex.from_tuples(self.F.index)

        elif self.level_of_detail in ['Link-1997 level', 'Detail level']:
            # dropping empty sectors (mostly Cannabis related)
            to_drop = concordance.loc[concordance.loc[:, 'IOIC'].isna()].index
            concordance.drop(to_drop, inplace=True)
            ghgs = pd.DataFrame()
            for code in concordance.index:
                # L97 and D levels are more precise than GHG accounts, we use market share to distribute GHGs
                sectors_to_split = [i[1] for i in self.industries if
                                    i[0] in concordance.loc[code].dropna().values.tolist()]
                output_sectors_to_split = self.V.loc[:,
                                          [i for i in self.V.columns if i[1] in sectors_to_split]].sum()
                share_sectors_to_split = pd.Series(0, output_sectors_to_split.index)
                for province in output_sectors_to_split.index.levels[0]:
                    share_sectors_to_split.loc[province] = ((output_sectors_to_split.loc[province] /
                                                             output_sectors_to_split.loc[province].sum()).fillna(
                        0).values) * GHG.loc(axis=0)[:, code].loc[province].iloc[0, 0]
                ghgs = pd.concat([ghgs, share_sectors_to_split])
            ghgs.index = pd.MultiIndex.from_tuples(ghgs.index)
            ghgs.columns = pd.MultiIndex.from_product([[''], ['GHGs'], ['Air']])

            # spatializing GHG emissions
            ghgs = pd.concat([ghgs] * len(ghgs.index.levels[0]), axis=1)
            ghgs.columns = pd.MultiIndex.from_product([self.matching_dict, ['GHGs'], ['Air']])
            for province in ghgs.columns.levels[0]:
                ghgs.loc[[i for i in ghgs.index.levels[0] if i != province], province] = 0
            # adding GHG accounts to pollutants
            self.F = pd.concat([self.F, ghgs.T])
            # reindexing
            self.F = self.F.reindex(self.U.columns, axis=1)

        # GHG emissions for households
        self.FY = pd.DataFrame(0, FD_GHG.columns, self.Y.columns)
        self.FY.update(FD_GHG.T)
        # spatializing them too
        self.FY = pd.concat([self.FY] * len(GHG.index.levels[0]))
        self.FY.index = pd.MultiIndex.from_product([self.matching_dict, ['GHGs'], ['Air']])
        for province in self.FY.columns.levels[0]:
            self.FY.loc[[i for i in self.FY.columns.levels[0] if i != province], province] = 0

        self.emission_metadata.loc['GHGs', 'CAS Number'] = 'N/A'
        self.emission_metadata.loc['GHGs', 'Unit'] = 'kgCO2eq'


def todf(data):
    """ Simple function to inspect pyomo element as Pandas DataFrame"""
    try:
        out = pd.Series(data.get_values())
    except AttributeError:
        # probably already is a dataframe
        out = data

    if out.index.nlevels > 1:
        out = out.unstack()
    return out


# pyomo optimization functions
def reconcile_one_product_market(uy, u0, imp, penalty_multiplicator):
    opt = SolverFactory('ipopt')

    # Define model and parameter
    model = ConcreteModel()
    model.U0 = u0
    model.UY = uy
    model.imports = imp

    # Large number used as penalty for slack in the objective function.
    # Defined here as a multiplicative of the largest import value in U0.
    # If solver gives a value error, can adjust penalty multiplicator.
    big = model.U0.max() * penalty_multiplicator

    # Define dimensions ("sets") over which to loop
    model.sectors = model.UY.index.to_list()
    model.non_null_sectors = model.U0[model.U0 != 0].index.to_list()

    # When defining our variable Ui, we initialize it close to U0, really gives the solver a break
    def initialize_close_to_U0(model, sector):
        return model.U0[sector]

    model.Ui = Var(model.sectors, domain=NonNegativeReals, initialize=initialize_close_to_U0)

    # Two slack variables to help our solvers reach a feasible solution
    model.slack_total = Var(domain=Reals)
    model.slack_positive = Var(model.sectors, domain=NonNegativeReals)

    # (soft) Constraint 1: (near) conservation of imports, linked to slack_total
    def cons_total(model):
        return sum(model.Ui[i] for i in model.sectors) + model.slack_total == model.imports

    model.constraint_total = Constraint(rule=cons_total)

    # (soft) Constraint 2: sectoral imports (nearly) always smaller than sectoral use
    def cons_positive(model, sector):
        return model.UY.loc[sector] - model.Ui[sector] >= - model.slack_positive[sector]

    model.constraint_positive = Constraint(model.sectors, rule=cons_positive)

    # Objective function
    def obj_minimize(model):
        # Penalty for relatively deviating from initial estimate _and_ for using slack variables
        # Note the use of big
        return sum(
            ((model.U0[sector] - model.Ui[sector]) / model.U0[sector]) ** 2 for sector in model.non_null_sectors) + \
               big * model.slack_total ** 2 + \
               big * sum(model.slack_positive[i] ** 2 for i in model.sectors)

    model.obj = Objective(rule=obj_minimize, sense=minimize)

    # Solve
    sol = opt.solve(model)
    return todf(model.Ui), model.slack_total.get_values()[None], todf(model.slack_positive)


def reconcile_entire_region(U_Y, initial_distribution, total_imports):
    # Dataframe to fill
    Ui = pd.DataFrame(dtype=float).reindex_like(U_Y)

    # Slack dataframes to, for the record
    S_imports = pd.Series(index=total_imports.index, dtype=float)
    S_positive = pd.DataFrame(dtype=float).reindex_like(U_Y)

    # Loop over all products, selecting the market
    for product in initial_distribution.index:
        uy = U_Y.loc[product]
        u0 = initial_distribution.loc[product]
        imp = total_imports[product]
        penalty_multiplicators = [1E10, 1E9, 1E8, 1E7, 1E6, 1E5, 1E4, 1E3, 1E3, 1E2, 10]

        # Loop through penalty functions until the solver (hopefully) succeeds
        for pen in penalty_multiplicators:
            try:
                ui, slack_import, slack_positive = reconcile_one_product_market(uy, u0, imp, pen)
            except ValueError as e:
                if pen == penalty_multiplicators[-1]:
                    raise e
            else:
                break

        # Assign the rebalanced imports to the right market row
        Ui.loc[product, :] = ui

        # commit slack values to history
        S_imports[product] = slack_import
        S_positive.loc[product, :] = slack_positive

    return Ui, S_imports, S_positive

