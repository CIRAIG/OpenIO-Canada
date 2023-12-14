## _OpenIO-Canada v2.8_

Python class creating Multi-Regional symmetric Environmentally Extended Input-Output (MREEIO) tables for Canada. OpenIO-Canada 
operates at the provincial level (13 provinces). It can thus be used to compare the environmental impacts of value chains
or consumption by households from any specific province.

OpenIO-Canada covers 492 commodities, 33 GHGs, 310 pollutants in 3 compartments (air, water and soil), 
67 mineral resources, water consumption and energy use.

OpenIO-Canada is connected to the Exiobase global MRIO database to model value chains happening outside Canada.

### Getting started

Clone the repository (or download it) and install the different libraries required to run this Python class (requirements.txt).<br>
Note that we recommend working with version **3.9.7** of Python as we can ensure it works for that specific version.<br>
Go to the doc folder and take a look at the demo.ipynb file to see how to generate the IO tables.

### Data used
- economic data: https://www150.statcan.gc.ca/n1/en/catalogue/15-602-X
- international merchandise trade data: https://open.canada.ca/data/en/dataset/b1126a07-fd85-4d56-8395-143aba1747a4
- main ghg emissions (CO2, CH4 and N2O): https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3810009701
- SF6, HFCs and PFCs: https://doi.org/10.5281/zenodo.7432088
- water use: https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3810025001
- energy use: https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3810009601
- other pollutants: https://open.canada.ca/data/en/dataset/1fb7d8d4-7713-4ec6-b957-4a882a84fed3
- mineral extractions: https://www.usgs.gov/centers/national-minerals-information-center/latin-america-and-canada-north-america-central-america#ca
- exiobase3: https://doi.org/10.5281/zenodo.5589597
- exiobase3 capital endogenization: https://doi.org/10.5281/zenodo.7073276
- IMPACT World+ impact assessment methodology: https://doi.org/10.5281/zenodo.5669648


### Miscellaneous information
Support for the industry version (_ixi_) is currently discontinued as it requires too much work. Support for lower detail 
IOIC classifications (i.e., Summary, Link-1961 and Link-1997) is also discontinued.

An article describing the methodology and results of openIO-Canada will be published soon.