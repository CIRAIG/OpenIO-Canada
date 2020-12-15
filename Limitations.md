The following document present the limitations of open IO Canada v2.0.

- GHG flows are only available from Statistics Canada as aggregated GWP flows, characterized using IPCC2007 factors
- Primary energy flows from Statistics Canada are only available as aggregated primary energy and already includes 
electricity which would trigger a double counting with economic electricity flows. Hence, energy use is not available 
for now.
- Water use flows are only available per sector at the national level. The output of each province for each sector was
used to distribute water use flows across provinces. For households demands in water, the consumption levels of each 
provinces were used to distribute households' water flow uses across each province.
- Inter-provincial trade were allocated using provincial outputs, i.e., the amount of aluminium imported by Quebec from 
other provinces was allocated according to the output of each province.These inter-provincial trade were then optimized 
using pyomo to avoid the appearance of negative entries
- International trade is not endogenized and as such is considered produced the same way as in the consuming province,
e.g., aluminium imported by Quebec is considered produced as in Quebec (even if it would normally come from China)
- NPRI emissions from the Education services sector were (arbitrarily) split 50/50 between the public and private sector
- There is a non negligible amount of emissions from the NPRI that could not be matched to the IMPACT World+ impact 
assessment methodology, hence having a null impact.