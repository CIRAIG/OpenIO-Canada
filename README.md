## _OpenIO-Canada v2.4_

Python class creating Multi-Regional symmetric Environmentally Extended Input-Output (MREEIO) tables for Canada. OpenIO-Canada 
operates at the provincial level (13 provinces). It can thus be used to compare the environmental impacts of value chains
or consumption by households from any specific province.

OpenIO-Canada covers 492 commodities, 310 pollutants including 3 greenhouse gases (CO2, CH4 and N2O) in 3 compartments 
(air, water and soil), water use and energy use.

OpenIO-Canada is connected to the Exiobase global MRIO database to model value chains happening outside Canada.

### Getting started

Clone the repository (or download it) and install the different libraries required to run this Python class (requirements.txt).
Go to the doc folder and take a look at the demo.ipynb file to see how to generate the IO tables.

### Data used
- economic data: https://www150.statcan.gc.ca/n1/en/catalogue/15-602-X
- international merchandise trade data: https://open.canada.ca/data/en/dataset/b1126a07-fd85-4d56-8395-143aba1747a4
- ghg emissions: https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3810009701
- water use:https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3810025001
- energy use: https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3810009601
- other pollutants: https://open.canada.ca/data/en/dataset/1fb7d8d4-7713-4ec6-b957-4a882a84fed3
- exiobase3: https://doi.org/10.5281/zenodo.5589597
- The IMPACT World+ impact assessment methodology is used


### Miscellaneous information
This project was in part funded by Shared Services Canada (SSC) but they are not responsible for any data/results 
obtained from open IO Canada.

Support for the industry version (_ixi_) is currently discontinued as it requires too much work. Support for lower detail 
IOIC classifications (i.e., Summary, Link-1961 and Link-1997) is also discontinued.

A scientific article describing the methodology and results of openIO-Canada is being written.