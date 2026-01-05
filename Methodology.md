This document outlines the methodology for the openIO-Canada model (updated for v2.11). The model is designed to provide
a comprehensive view of the Canadian economy and its environmental impacts by integrating national economic data with 
global trade and environmental accounts.

## 1. Economic Modeling

The core of openIO-Canada is built upon Supply and Use tables (SUTs) sourced directly from Statistics Canada 
(https://www150.statcan.gc.ca/n1/en/catalogue/15-602-X).

### Data granularity
Statistics Canada provides SUTs at four levels of detail: Summary, Link-1961, Link-1997, and Detail. 
OpenIO-Canada utilizes the Detail level, which provides a granular view of the economy including:
- 492 commodities.
- 240 industries.
- 279 final demand categories.
- 8 factors of production (such as salaries and taxes).

The model operates exclusively in basic prices. To achieve this, it uses trade, transport, and tax details provided in 
the SUTs to convert data from purchaser prices to basic prices. Data is included for all 13 provinces and territories, 
while Canadian enclaves (embassies) are excluded due to a lack of emission data.

### Endogenizing Capitals
As a reminder, the endogenization of capitals consists in attributing the use of capitals to the sectors of the economy 
that are using them. If you are familiar with LCA, endogenization is the equivalent of putting a 1e-14 factory in your 
LCA process. There are two main approaches to endogenization of capitals: the augmentation method and the flow matrix method. 
The first, endogenizes capitals into a single aggregated new sector of the intermediate economy (called it “capital” 
if you will), while the second disaggregates the different capital uses to the actors of the economy, so in essence it 
results in a capital matrix (K) that has the same dimensions as the technology matrix. In openIO-Canada we have always 
followed the flow matrix method.

Now what capitals do we endogenize? There are two metrics here: the gross fixed capital formation and the consumption of
fixed capitals. What is the difference? Gross fixed capital formation are the capital goods that were purchased (and 
then assumed built) this year. Consumption of fixed capital represents the value of capital goods that became 
obsolete/damaged that had to be replaced this year. Literature typically recommends endogenizing the consumption of 
fixed capitals, since this better represents existing capital goods, that were used to produce any commodity this year. 
Indeed, the factory that was just built this year is most likely not in operation yet. In openIO-Canada, since the v2.11,
we now follow these recommendations and also endogenize the consumption of fixed capitals. In previous versions, the 
gross fixed capital formation was endogenized, due to a lack of data. 

So concretely how is it done? The first step is to determine what is the consumption of fixed capitals for each sector 
of the economy. This information is not directly available within the supply and use tables of Statistics Canada. Indeed,
we only have access to “gross mixed income” and to “gross operating surplus”, both of which include the consumption of 
fixed capital, along with various other things, such as direct profits of the sectors. The only information we have
access to is another source of data from Statistics Canada which provides details on the factors of production for Canada
and all its provinces. Among these factors of production, we can find values for “Consumption of fixed capital: 
corporations”, “Consumption of fixed capital: unincorporated businesses” and “Consumption of fixed capital: general 
governments and non-profit institutions serving households”. However, these values are not available per sectors of 
the economy. In other words, through this data we know that, e.g., Ontario has a consumption of fixed capitals of ~76 
billion of dollars in 2021, meaning that this amount of building had to be rebuilt due to the use of capital goods over
the years. However, we do not know if the capital that had to be rebuilt was a hospital, a road, a software platform, 
a museum, etc.

To estimate to which capital goods the consumption refers to, we derive historical gross fixed capital formation values,
meaning that we simply take the final demands of the supply and use tables of StatCan, for years 2014 to 2021 and make 
a distribution out of them. So if over this 8 years-period on average 1% of the capital built was hospitals in Quebec, 
then we will assume that 1% of the consumption of fixed capital in Quebec pertains to hospitals. Ideally, a longer 
period than 2014-2021 should be covered, as the lifetime of some capitals (mainly buildings) far exceeds this 8 
year-period. However, the data only goes back up to 2009 and includes a lot of changes in classifications which require 
a lot of work to go through. We thus assume that this 8 year-period is representative enough.

Being able to determine which capital goods were consumed, and thus which sectors of the economy consumed these capital 
goods (through a simple mapping), we can now construct the capital matrix (K), which is then gradually transformed into 
a pxp matrix, with the same dimensions than the technology matrix (A), by passing through the same steps as said 
technology matrix.

One final operation is needed however. Indeed, now that the capital consumed have been endogenized, these endogenized 
capitals only represent the capital goods that needed to be replaced. However, the economy is growing, and new capital 
goods are constructed, simply to match the growing demand. These new capital goods also need to be included in the 
yearly assessments. We thus calculate these newly built capital goods, by subtracting the consumed capital goods from 
the total gross fixed capital formation. In doing so, the final demand matrix (Y) of the endogenized system still 
includes the formation of capital goods, but that only pertains to newly built capital goods, and not replaced ones.

### Interprovincial trade
The interprovincial trade is detailed in Statistics Canada Supply and Use tables and describes what each province is 
importing from which province. However, it does not inform on the final destination of imports. In other words, the data
shows that Quebec imported gasoline from Ontario, but we do not know if the gasoline was consumed by sector X or Y or 
even by the households. We therefore redistribute imports proportionate to the use of the commodity by a given consumer 
in the economy. For instance, if households account for 25% of the purchase of gasoline within the Quebec province, 
then 25% of the imported Ontarian gasoline will be attributed to the households.

This logic to distribute interprovincial imports is also used for capital goods (in the case of endogenization). Hence, 
if households are responsible for 50% of the car purchases in Quebec, then they would inherit of 50% of car imports.

There are few cases where the import and use data do not match and we end up with provinces importing more than what is 
used that current year. In that case, the excess import is transferred to the final demand sector “Changes in inventories,
finished goods and goods in process”. This only happens for territories with very small economies, such as Yukon, 
Northwest territories and Nunavut.

### International trade
The supply table of Statistics Canada provides international import values for all commodities in each province 
(contained in the INT_imports table of openIO-Canada). However, it crucially lacks two information. First, the final 
destination of the import is unknown, just like in the case of interprovincial imports. Second, the country origin of 
the import is also unknown (so we don’t know if an imported car in Ontario comes from Germany, the US or China). 
OpenIO-Canada thus relies on the merchandise trade database from Statistics Canada to determine the average origin of 
imported commodities, for each province. This trade database relies on the Harmonized System (HS) classification and we 
specifically use the HS-6 level of the classification which is much more detailed than the detail level classification 
of openIO-Canada. The mapping between commodities can be found in the HS-IOIC.xlsx file. Thus, each commodity of openIO-Canada is mapped
to a commodity of the HS-6 classification; for each province we then have the average origin of imports of the 
commodity; the obtained geographical distribution is then translated to regions from the exiobase database. Note that 
the merchandise trade database is only used to provide a geographical distribution of origin and NOT the values of 
imported goods. For the latter, we still rely on the supply and use tables.


Once the region of origin of each import is determined, we determine the destination of the imports with the same method 
as the interprovincial imports, i.e., proportionally to the use of the imported commodity. This results in import 
matrices matching with openIO-Canada’s classification (merchandise_imports_scaled_U, merchandise_imports_scaled_K 
and merchandise_imports_scaled_Y).

### Symmetric table construction

The model applies the **industry technology construct** to derive symmetric product-by-product matrices. The core 
technology matrix $A$ is calculated as:

A = U · ĝ⁻¹ · Vᵀ · q̂⁻¹

Where:
* U is the use table.
* Vᵀ is the transposed supply table.
* ĝ⁻¹ and q̂⁻¹ are diagonalized, inverted industry and product output vectors.

The factors of production matrix (R) and the capital matrix (K) undergo similar transformations. To preserve financial 
balance, the following equation must hold for every product x:

1 = Ax + Rx + Wx

### Link with exiobase
With international imports in hand, we link these imports to the Global MRIO database exiobase to determine the 
respective environmental impacts of each imported commodity. Since we endogenized the capitals in openIO-Canada, we use 
the endogenized version of exiobase to keep consistency.

The mapping between openIO-Canada and exiobase classifications can be found in the IOIC_EXIOBASE.xlsx file. In the end, 
the link consists of connecting an import matrix between openIO-Canada and exiobase.

Data in the international imports matrix are converted to €, using the conversion rate of the given year (e.g., 
1.4856$CAD/€ in 2019). Data from exiobase are divided by 1000000 to obtain /€ data (by default exiobase is in /million €).

Note that openIO-Canada feeds on exiobase to improve its coverage, but exiobase within the openIO-Canada framework does 
not use the Canadian economy from openIO-Canada. In other words, the US economy of exiobase still imports Canadian 
products from the Canadian economy as described by exiobase. It also means that within the life cycle of Canadian 
commodities of openIO-Canada, there will be “international imports” from Canada (from exiobase), because e.g., 
the imported US car imported steel from the Canadian economy of exiobase to be manufactured.

The merchandise trade database as its name suggests covers the trade of merchandise but does not cover the import of 
services. There is no specific service trade database and we thus reuse the data of exiobase to represent the 
international imports of services in openIO-Canada. Exiobase is only used to define the relative share of countries 
for a given import category. We keep the values of imported services provided by Statistics Canada in the supply and 
use tables.

Once the link between openIO-Canada and exiobase is done, we obtain concatenated matrices englobing both databases. 
For instance, we get a total symmetric technology matrix A including the technology matrices of openIO-Canada and 
exiobase and also including the connection matrix between the two systems, representing by the international imports. 
Similar concatenated matrices were created for final demand (Y) and capital flow matrix (K).

## 2. Environmental Modeling

#### Main GHGs (CO2, CH4, N2O)
The three main GHGs emission come from the GHG physical accounts of Statistics Canada. These accounts follow the System 
of Environmental-Economic Accounting (SEEA) framework. The public release of these accounts provide aggregated GHG 
emissions of these 3 GHGs in CO2 equivalents for 117 industries. However, on demand we could obtain a disaggregated 
version from Statistics Canada, where the emissions of CO2, CH4 and N2O are provided. We could then associate up-to-date
GWP100 characterization factors (namely IPCC AR6 values), but also connect to other impact categories such as marine 
acidification or the impact of climate change on human health and biodiversity.

Since the GHG emissions are only provided by a more aggregated level than the classification of openIO-Canada (GHGs for 
115 industries vs 240 industries with the detail level of supply and use accounts), the GHG emissions were distributed 
proportionate to the sales. For example, if “Wheat” represents 5% of the sales of “Crop production” then “Wheat” gets 
5% of all CO2, CH4 and N2O emissions of “Crop production”.

Physical accounts also provide the GHG emissions for households, which they separate in two categories: “Motor fuels and
lubricants”, which mainly pertains to emissions of using one’s car and “Heating, lighting and appliances” which 
corresponds to burning fuels at home. These were linked to two final demand household categories. It was impossible 
to redistribute emissions for “Heating, lighting and appliances” between the two household categories “PEC04520 - Gas”
and “PEC045A0 – Other fuels”. Therefore, we took the arbitrary choice of attributing all “Heating, lighting and 
appliances” GHG emissions to the “PEC045A0 – Other fuels” consumption category. Which means that if a user looks at the 
contribution of the consumption category “PEC04520 – Gas” within households’ carbon footprints, it does not include the 
combustion of gas burned at home, since the latter is aggregated with the combustion of other fuels at home.

#### Refining GHG emissions
There are two main limitations of the physical accounts provided by Statistics Canada which we tackled in openIO-Canada.
The first one is the fact that biogenic carbon and fossil carbon are not differentiated. Not making the difference 
obviously results in an overestimation of actual GHG emissions. To correct this issue, we rely on the relative share of
biogenic vs fossil carbon in exiobase sectors. If exiobase displays a 10% biogenic CO2 release for sector X, then we 
assume the same biogenic ratio for sectors of openIO-Canada connected to X (either included in or including X). This 
measure should correct to a satisfactory level the issue for the industry. However, there are no biogenic emissions 
within the final demand emissions of exiobase. We were thus unable to apply the same logic for the final demand GHG 
emissions of openIO-Canada. The latter thus were not further corrected and biogenic carbon released by the final demand 
is thus considered as fossil, which corresponds to an estimation. This is a clearly a limit of the current openIO-Canada 
database, especially accounting for the fact that many Canadians rely on biomass for heating at home.

The second issue is the fact that the agricultural sector is extremely aggregated in the physical accounts. The GHG 
emissions are only available for the very broad “Crop and animal production (except cannabis)” sector. Applying the 
economic allocation as openIO-Canada is doing thus results in attributing an increased share of direct GHG emissions to 
crops rather than to animal feedstock. This is clearly an issue. To deal with this problem we once again rely on exiobase
to separate GHG emissions between crops and animals. We determine the ratios of GHG emissions for crop-related sectors 
in exiobase (e.g., Wheat, Cereal grains nec, etc.) and meat-related sectors (e.g., Cattle, Pigs, etc.) and use these 
ratios to separate the emissions of the very broad “Crop and animal production (except cannabis)” sector into emissions 
for “Crop production (except cannabis, greenhouse, nursery and floriculture production)”, “Greenhouse, nursery and 
floriculture production (except cannabis)”, “Animal production (except aquaculture)” and “Aquaculture”. More work on 
direct emissions of agriculture and especially feedstock will be necessary in future versions of openIO-Canada to ensure
a better modeling of these crucial sectors.

#### Covering more GHGs
The physical accounts only provide information for the three main GHGs. OpenIO-Canada thus relies on another data source
to cover additional GHGs: the UNFCCC database. The latter is used to mostly add SF6, HFCs and PFCs, resulting in the 
coverage of overall 36 GHGs (including CO2, CH4 and N2O). Note that the use of the UNFCCC database is inconsistent with 
the Input-Output framework. Indeed, the latter relies on the SEEA (System of Environmental-Economic Accounting) 
framework. This entails some differences in the way of accounting for carbon, such as, whenever a plane operated by a 
Canadian company (e.g., Ari Canada) departs from Paris to land in Montreal, to which country are the GHG emissions 
attributed? France or Canada? Each framework has a slightly different interpretation. However, we believe that relying 
on these slightly different account rules to cover more GHGs is still closer to “the truth” that not covering these 
additional GHGs at all.

#### Water consumption
Water consumption data is not directly provided by StatCan physical flow account. What the latter are providing is water
use. What is the difference you may ask. Well water use is simply the fact of using the water. So, if in my factory I 
use 1,000m3 of water for cooling, this water does not disappear entirely. We could say that maybe 5m3 of water will 
evaporate and 9,995m3 of water will be re-used later in the factory, or released in the environment, or sent for water 
treatment which will redistribute water later on. So here, the water usage of the factory is 1,000m3, but the water 
consumption is actually only 5m3. This is important as using water is not detrimental to biodiversity (well except if we
were taking into consideration the decrease in quality of the water, but this is not considered in life cycle impact 
assessment methods for now). What is detrimental is making water unavailable for biodiversity, or for human consumption.

So, we first rely on water usage from the physical flow accounts for water use of StatCan, BUT we will apply consumption
ratio estimates, depending on sectors of the economy. For instance, literature estimates to around 10% the amount of 
water that is consumed by citizens using water (washing your car or watering your garden). Another example, in the case 
of mining, this number goes up to 20%.

Three notable exceptions for three sectors for which we obtain direct water consumption figures, instead of applying a 
consumption ratio to water use data:
- Crops: for crops we rely on Pfister et al. (2011) [https://dx.doi.org/10.1021/es1041755]. A mapping between the 
commodities described in Pfister et al. and the crop commodities in openIO-Canada was made, to be able to obtain total 
water consumption in a given for a given set of crops.
- Animal feedstocks: Water consumption per animal type was obtained from a tool from the Agriculture ministry of the 
Albertan provincial government (www.agric.gov.ab.ca/app19/calc/livestock/waterreq_dataentry2.jsp). This data source 
provides consumption figure in L/day/animal and these values were thus scaled up to match the total number of animals 
bred in each province from various Statistics Canada tables.
- Electricity: As can be surmised, the electricity sector is a big consumer of water, especially in Canada, through the 
use of hydro-electricity (notably in Québec) but also through the use of nuclear energy (in Ontario for example). 
Water use data only being available for the electricity sector overall was an issue and so water consumption data per 
electricity generation technology were obtained. We used values from the same study from Pfister et al. which provided 
water consumption numbers for technologies except for hydro-electricity, although those values were determined for the 
US, Switzerland and Tanzania and so not adapted for Canada. For the hydro-electricity, we relied on the 2012 technical 
report version 1 of the Quantis Water Database which estimated to 14,7 m3/MWh the consumption of water for 
hydro-electricity. We then scaled up consumption figures using the installed MWh capacities per technology in the 
different provinces.

In addition, water consumed in Canada does not have the same impact than water consumed in Saudi Arabia. To account for 
the difference in scarcity of the water, we need to spatialize the water flows, that is, to assign a location to them 
(water extracted in Canada vs in Saudi Arabia). This spatialization is done on water flows of openIO-Canada (thus 
differentiating between water extracted in Québec vs British Columbia) but also on Exiobase. This means that the 
Exiobase version provided through openIO-Canada contains spatialized water flows.

#### Energy consumption
Energy use data comes from the physical flow accounts for energy use of Statistics Canada. This data is at a more 
aggregated level than the detailed level of classification used in openIO-Canada, so a similar mapping and distribution 
rule based on market share was applied.

#### Mineral extraction
Mineral extraction data comes from the USGS (United States Geological Survey) database. A manual extraction of data for 
67 minerals/metals from Excel files of the USGS database was performed and the data gathered in the 
Minerals_extracted_in_Canada.xlsx Excel file from openIO-Canada. These minerals/metals were then mapped to the various 
mining sectors of openIO-Canada (see concordance_metals.json for the mapping).

#### National Pollutant Release Inventory (NPRI)
Emissions for all other pollutants come from the National Pollutant Release Inventory (NPRI) database of ECCC 
(Environment and Climate Change Canada). This database contains information that big emitting industries have to provide
the government. This therefore contains information at the factory level. However, in the case of openIO-Canada, the 
information is simply aggregated at the province and sectoral level. The data is by no means complete. First, only the 
big emitters are technically obligated to provide emissions, meaning that most medium to small factories are excluded 
from the data. Furthermore, even factories technically obligated to provide information do not reliably do so, and when 
they do, perhaps not with the best precision let’s say. In other words, while openIO-Canada provides estimates for these
pollutants, it needs to be interpreted as being a significant underestimation of the actual emissions.

#### Plastic waste emissions
Plastic waste emissions are based on the corresponding physical flow account (PPFA) from Statistics Canada  for plastic 
waste managed in Canada and on OECD’s “Plastic waste by region and end-of-life fate” data  for plastic waste managed in 
the rest of the world. 

As its name indicates, the PPFA covers the plastic waste managed within Canada, with (sometimes) a provincial coverage. 
It provides 17 indicators on plastic waste, among which we selected 4 indicators to include in openIO-Canada. These are: 
“Disposed plastic waste and scrap sent to landfill or incinerated without energy recovery” which represents the amount 
of plastic waste that is eventually sent to landfill and incineration without energy recovery, “Disposed plastic waste 
and scrap sent for incineration or gasification with energy recovery” which represents the amount of plastic waste that 
is eventually sent to incineration and gasification with energy recovery, “Recycled plastic pellets and flakes ready for
use in production of new products or chemicals” which represents the amount of plastic waste that is eventually actually
recycled into new products and finally “Plastic leaked permanently into the environment” which represents the amount of 
plastic waste that is ultimately permanently lost into the environment. These 4 indicators are available through two 
files. The first one details the plastic waste management for product categories (e.g., construction materials, 
electronics, vehicles, etc.), the second one describes the plastic waste management per resin type of plastics (e.g., 
PET, HDPE, PVC, PS, etc.). Within openIO-Canada, both files are used. The first connect openIO-Canada with the 
“per-product-category” file and then connect the “per-resin” file to the “per-product-category” file, hence linking it 
to openIO-Canada as well. Below we describe the process and assumptions made to integrate these files to the 
openIO-Canada framework.

OECD’s “Plastic waste by region and end-of-life fate” data describes 5 fates for plastic end-of-life: “Recycled”, 
“Landfilled”, “Incinerated”, “Littered” and “Mismanaged” for 7 OECD and non-OECD regions as well as 4 specific 
countries: USA, Canada, China and India. 

_How do we link these end-of-life management data to production data in openIO-Canada?_

The first step is to complete the “per-product-category” file of the PPFA for missing information. Some product 
categories are not available per province and we extrapolate their value from higher-level category and the national 
ratio for sub-categories. In other words, we have provincial information for “Packaging” but not for “Bottles” and 
“Films”. So, we extrapolate the values of “Bottles” and “Films” at provincial level, by calculating the ratios of these 
two products within the broader “Packaging” category. We then apply this ratio to the “Packaging” values for each 
province. Conversely, some relevant indicators for plastic waste management are only defined at national level. Using a 
similar approach, we estimate the provincial values for “Disposed plastic waste and scrap sent to landfill or 
incinerated without energy recovery” and “Disposed plastic waste and scrap sent for incineration or gasification with 
energy recovery” using the higher-level category “Total disposed plastic waste and scrap”.

With data gaps in the PPFA filled, we now need to determine where plastics are actually embedded. Indeed, the PPFA only 
provides information on where the plastics ends up. But to link this information to openIO-Canada, we need to determine 
the origin, in other words, which sector of the economy it comes from. Otherwise, we could link every plastic waste data
to the relevant waste management sector, but that provides little to no information. To determine where the plastic is 
embedded, we use proxies that should reliably inform on the presence or not of plastic embedded in a product. The two 
sectors used as proxies are “Plastic resin” for the Canadian economy and “Plastics, basic” for international imports. 
Since these two sectors represent purchases of basic plastic (e.g., monomers and resins), we assume that their only use 
is to be incorporated within the final product. It is an assumption, as companies might purchase resins to produce their
own packaging, which thus does not end up in the final product, but in the packaging.

We thus use the Z matrix to determine the sectors that bought plastic resins. To avoid accounting for almost every 
commodity in the database (with some 1e-13$ of plastic resins bought) we cover ~95% of the total of traded plastic 
resins. We then link these identified commodities to the PPFA categories such as, e.g., the PPFA category “Vehicles” 
is linked to the openIO-Canada commodities “Motor vehicle plastic parts”, “Motor vehicle interior trim, seats and seat 
parts”, “Motor vehicle electrical and electronic equipment”, ‘Tires” and “Other miscellaneous motor vehicle parts”. 
For each of these commodities, we calculate the market of that product: local production, interprovincial imports and 
international imports. That way, we are able to link a disposed product embedding plastic (a plastic waste from vehicle
ending its life in Yukon), to its typical origin (not produced in Yukon clearly). Each of the identified commodities 
however, embed a different quantity of plastic/€. There is probably a bigger % of composition of plastic in “Motor 
vehicle plastic parts” than in “Motor vehicle interior trim, seats and seat parts”. So, we need to convert the purchases
of each of the commodities into purchases of embedded plastic. We once again rely on proxies: “Plastic resins” and 
“Plastics, basic”. We end up with a distribution of where plastics is embedded for products of category. For instance, 
for a “Vehicles” in Quebec, plastic is typically embedded 48% in “Motor vehicle plastic parts”, 47% in “Tires”, 4% in 
“Motor vehicle interior trim, seats and seat parts” and almost nothing in the two last commodities.

Using this newly gained distribution, along with the market distribution of the product we can attribute a portion of 
the plastic waste management data (e.g., X tonnes of plastic recycled in Quebec for the “Vehicles” category) to 
different sectors of the global economy (because in our market distribution we included interprovincial and 
international imports).

We just dealt with the data from the PPFA. Notice that in the end we distributed the plastic waste data of the PPFA 
through the whole global economy, not just to Canadian sectors. We will do a similar process with the OECD data. 
Similarly to what was described previously, we first determine where the plastic is typically embodied, i.e., in which 
products across the global economy can plastic be found. We use the “Plastics, basic” sector as a proxy, assuming that 
if a sector purchases “Plastics, basic”, these resins and base of plastic will be incorporated in the outgoing products 
of the sector. We rely on the Z matrix of exiobase to determine the consumption market of “Plastics, basic” across the 
world. For example, in 2019 the sector (“CN”, “Rubber and plastic products (25)”) represented 24.3% of the total amount 
bought of “Plastics, basic”.

Once we know into which products are plastics embedded, we determine how much of these identified products a specific 
country is purchasing (e.g., all US sectors bought in total ~218,105,000,000€ of (“US”, “Rubber and plastic products 
(25)”) and ~13,600,000,000€ of (“CN”, “Rubber and plastic products (25)”) and so on). We then translate these monetary 
values of product goods embedding plastic into monetary values of embedded plastic directly, using once again “Plastics,
basic” as the proxy. The obtained distribution thus finally provides a way to allocate the plastic waste managed in a 
given country to where it probably originated from. Plastic waste management data from OECD is then simply applied with
the distribution.

For countries that are described in exiobase but not in the OECD data, we have to add an extra step is distributing the 
OECD data to the different countries of the region. For example, the OECD data only describes the plastic waste 
management of the OECD EU region, but not specifically of France, Germany, etc. So we need another proxy to be able to 
distribute the total plastic waste managed in a given region to the countries that constitute this region. To do so, 
we rely on the “Plastic waste for treatment” sectors of exiobase. The idea is that a country within an OECD region 
should be allocated more of the plastic waste managed if they have a plastic waste sector that make mores money. 
This of course is an assumption that disregards the fact that the plastic waste sectors in the different countries 
making a region do not operate at the same costs. In the end, we calculate three distributions using “Plastic waste for 
treatment: landfill”, Plastic waste for treatment: incineration” and the sum of these two and use these 3 distributions 
to distributes the different OECD plastic waste management data to countries within a region.

##### Specifying plastic resin compositions
Because the aim in openIO-Canada is not only to quantify the end-of-life management of plastic waste, but also to 
determine the impact on biodiversity of plastic waste lost to the environment, we need to add plastic resin compositions
information. Indeed, plastic resins will impact biodiversity in different ways. So, we determined average plastic 
compositions to link with plastic physical flow account information, per product category. In other words, what are the 
resins that are typically used in “Construction materials”, “Textiles”, etc. For plastic waste covered by the OECD data 
(which does not differentiate between product categories) we rely on plastic production data from several studies 
(links below). The ratios are simply applied to the plastic waste management data already linked in openIO-Canada. By 
doing so, we assume that each plastic resin has the same pattern of waste management, which is incorrect (PET is easier 
to recycle than expanded PS for instance). However, in this first version of the plastic extension, we believe it is an 
acceptable assumption.

##### Resin composition data source:
For resin composition per product category: based on a literature review of multiple studies. For further information on
that aspect, contact maxime.agez@polymtl.ca.
For Europe: Plastics Europe, the fast facts, [European plastics production – polymer figure] 
(https://plasticseurope.org/knowledge-hub/plastics-the-fast-facts-2023/)
For the US: Figure 2 of https://doi.org/10.1016/j.resconrec.2021.105440
For China: Figure 2 of https://doi.org/10.1016/j.resconrec.2023.106880
For India: Figure 4 of https://doi.org/10.3390/su14084425
For other countries/regions: Plastics Europe, the fast facts, [Global plastics production – polymer figure] 
(https://plasticseurope.org/knowledge-hub/plastics-the-fast-facts-2023/)

##### Impact of plastic waste onto biodiversity
The plastic waste that ends up lost to the environment will ultimately have an impact on biodiversity through its 
physical properties (think of microplastics in the lungs of fishes for instance). This plastic could impact biodiversity
in many environmental compartments, i.e., soil, air, freshwater and ocean. We rely on the work of the MariLCA group to 
quantify the physical effects of plastic on biota. For now, their work only covers impacts happening in the ocean. So we
need to determine how much of the plastic leaked permanently to the environment ends up in oceans, compared to how much 
ends up in rivers and soil. We rely on data from the plastic footprint network which estimates that on average 15% of 
medium-sized plastic particles end up in the oceans. This amount of plastic in the ocean is then connected to the 
corresponding characterization factors in the IMPACT World+ methodology (see Characterization factors section).

##### Validation of the plastic waste emissions
To attempt to validate obtained results, we compare the total amount of plastic waste generated by the average Canadian 
according to our model (~163kg in 2019 / openIO-Canada v2.9) to the literature.
- In Europe, in 2014, the amount of plastic waste generated was estimated to be between 90kg and 130kg 
(https://doi.org/10.1016/j.resconrec.2021.106086)
- In Europe again, in 2016, another study estimated the amount to be ~140kg (73Mt for 510 million people) 
(https://doi.org/10.1016/j.cesys.2020.100004)
- In Austria, in 2010, the amount of plastic products consumed per capita was estimated at 156kg 
(https://doi.org/10.1016/j.resconrec.2016.10.017)
- In Denmark, a technical report estimated the total amount of consumed plastics per capita in 2016 was ~170kg 
(130,000+840,000 for 5.7 million Danes) (Pivnenko, K., Damgaard, A., & Astrup, T. F. (Eds.) (2019). Preliminary 
assessment of plastic material flows in Denmark - Technical Report. Danish Environmental Protection Agency. 
Miljoeprojekter No. 2090)
- In Switzerland, the amount of plastics leaked to the environment was estimated to be between 440g and 760g per person 
in 2014. In openIO-Canada this amount is estimated to be ~800g for Canadians for 2019. (https://doi.org/10.1021/acs.est.9b02900)

## 3. Life cycle impact modeling
OpenIO-Canada relies on the IMPACT World+ life cycle impact assessment methodology to convert emissions in the
environment to potential impacts on the environment. The methodology of this method can be viewed here: 
https://github.com/CIRAIG/IWP_Reborn/tree/master/Methodology. Note that the team behind IMPACT World+ is the same team 
behind openIO-Canada.