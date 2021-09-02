The following document present the limitations of open IO Canada v2.2.

- GHG flows are only available for the L61 classification level. More disagreggared classification therefore allocate
emissions based on sales' volume within a broader sector.
- Primary energy flows from Statistics Canada are only available as aggregated primary energy and already includes 
electricity which would trigger a double counting with economic electricity flows. Hence, energy use is not available 
for now.
- Water use flows are only available per sector at the national level. The output of each province for each sector was
used to distribute water use flows across provinces. For households demands in water, the consumption levels of each 
provinces were used to distribute households' water flow uses across each province.
- Inter-provincial trade were allocated using provincial outputs, i.e., the amount of aluminium imported by Quebec from 
other provinces was allocated according to the output of each province.These inter-provincial trade were then optimized 
using pyomo to avoid the appearance of negative entries
- Impacts from international trade are estimated based on Exiobase. The origin of products though, is not provided by
Statistique Canada. In the end we only know that Quebec imports X kg of Beef. We do not know if it comes from the US or 
Mexico. So we use global sales volumes of countries to allocate international imports. In other words, if Mexico 
represents 12% of total sales of beef worldwide (excluding Canada itself), then 12% of the international beef imported 
from outside Canada will be considered coming from Mexico.
- NPRI emissions from the Education services sector were (arbitrarily) split 50/50 between the public and private sector
- There is a non negligible amount of emissions from the NPRI that could not be matched to the IMPACT World+ impact 
assessment methodology, hence having a null impact.