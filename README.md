**OpenIO Canada**

**_In development, not ready for use._**

Class that creates symmetric Input-Output tables based on the Supply and Use economic tables provided by Statistic
Canada (https://www150.statcan.gc.ca/n1/en/catalogue/15-602-X), the NPRI of Canada 
(https://open.canada.ca/data/en/dataset/1fb7d8d4-7713-4ec6-b957-4a882a84fed3)
and the physical flows accounts of GHG emissions, water use and energy use of Canada 
(https://www150.statcan.gc.ca/n1/tbl/csv/38100097-eng.zip).

Currently, the scope of the project only includes the creation of a system at the national scale. Ultimately, this 
system might be regionalized to differentiate trade and impacts linked to provincial trade, i.e., specific data for 
Quebec or Ontario would be available.

Tables are available both in _ixi_ and _pxp_ formats, generated either using the fixed industry sales structure or the industry technology 
assumptions. More transformation models might be added in the future.

Unfortunately, GHG emissions are not disaggregated and were precompiled using IPCC2007 impact factors, which are 
outdated. 

OpenIO includes 280 pollutants in 3 compartments (air, water and soil). These flows are linked to the IMPACT World+ 
life cycle impact assessment methodology.