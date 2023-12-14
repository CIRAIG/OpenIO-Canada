The following paragraphs present the limitations of openIO-Canada v2.8.

- The sectors within a province importing international goods is not known. In other words, we know Quebec is importing X$
amount of steel from the US, but we do not know if it's the construction or the manufacturing sector which needs the X$ 
of imported steel. Imports were thus distributed following the use structure within the province.
- CO, CH4 and N2O flows are only available for the L61 classification level. OpenIO-Canada uses a more disaggregated 
classification and therefore allocates emissions based on sales' volume within a broader sector.
- HFC/PFC/SF6 and NF3 emissions come from the UNFCCC data, whose guidelines differ from the SEEA.
- The physical flow accounts of GHG used in openIO-Canada follow the SEEA guidelines, which does not differentiate between
fossil and biogenic carbon. For intermediary exchanges, biogenic emissions level were estimated through Exiobase. For
final demand though, it could not be estimated through Exiobase. As a result, direct GHG emissions from households are 
overestimated, as it includes biogenic carbon emissions (from burning wood in the chimney or relying on biofuel for
one's car) and accounts for them as if they were fossil emissions.
- Some water flows are only available per sector at the national level. The output of each province for theses sectors was
used to distribute water use flows across provinces. For households demands in water, the consumption levels of each 
province were used to distribute households' water flow uses across each province.
- The NPRI only covers a handful of industrial sites. Pollutant emissions are therefore known to
currently be underestimated.
- NPRI emissions from the Education services sector were (arbitrarily) split 50/50 between the public and private sector
- There is a non-negligible amount of emissions from the NPRI that could not be matched to the IMPACT World+ impact 
assessment methodology, hence having a null impact.
- For the Canadian part of openIO-Canada, the capital formation (and not the consumption of capital) was endogenized.
- While the 2.8 version of openIO-Canada now covers water consumption instead of water use, the regionalization
of the associated impacts is not yet implemented.