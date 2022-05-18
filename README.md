**OpenIO-Canada v2.3**

Class that creates Multi-Regional symmetric Environmentally Extended Input-Output (EEIO) tables for Canada. OpenIO-Canada 
operates at the provincial level (13 provinces). It can thus be used to compare the environmental impacts of value chains from 
Quebec and Ontario for example.

Through the "Detail level" of economic data from Statistics Canada, openIO-Canada covers 492 commodities and 240 industries.

Covers 310 pollutants including 3 greenhouse gases (CO2, CH4 and N2O) in 3 compartments (air, water and soil). OpenIO-Canada 
also covers the use of water and energy.

The IMPACT World+ life cycle impact assessment methodology is used to characterize the impacts of emissions on the 
environment.

Tables are available both in _ixi_ (industry) and _pxp_ (product) formats.
The fixed industry sales structure assumption was used to generate the _ixi_ format and the industry technology 
assumption was used for the _pxp_ format. More transformation models might be added in the future.

Data used:
- economic data: https://www150.statcan.gc.ca/n1/en/catalogue/15-602-X
- ghg emissions: https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3810009701
- water use:https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3810025001
- energy use: https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3810009601
- other pollutants: https://open.canada.ca/data/en/dataset/1fb7d8d4-7713-4ec6-b957-4a882a84fed3

This project was in part funded by Shared Services Canada (SSC) but they are not responsible for any data/results 
obtained from open IO Canada.

An article describing the methodology of openIO-Canada is being written.