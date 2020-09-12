**OpenIO Canada**

Class that creates symmetric Input-Output tables based on the Supply and Use economic tables provided by Statistic
Canada (https://www150.statcan.gc.ca/n1/en/catalogue/15-602-X), the NPRI inventories 
(https://open.canada.ca/data/en/dataset/1fb7d8d4-7713-4ec6-b957-4a882a84fed3)
and the physical flows accounts of GHG emissions (https://www150.statcan.gc.ca/n1/tbl/csv/38100097-eng.zip).

Available both in _ixi_ and _pxp_ formats, generated using the fixed industry sales structure and industry technology 
assumptions, respectively. More transformation models might be added in the future.

Unfortunately, GHG emissions are not disaggregated. On the other hand, OpenIO follows 280 pollutants
in 2 compartments (air and water). These flows are linked to the IMPACT World+ life cycle impact assessment methodology.