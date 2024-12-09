## _OpenIO-Canada v2.10_

Python class creating Multi-Regional symmetric Environmentally Extended Input-Output (MREEIO) tables for Canada. OpenIO-Canada 
operates at the provincial level (13 provinces). It can thus be used to compare the environmental impacts of value chains
or consumption by households from any specific province.

The database covers reference years from 2017 to 2021.

OpenIO-Canada covers 492 commodities, 33 GHGs, 310 pollutants in 3 compartments (air, water and soil), 
67 mineral resources, water consumption, energy use and plastic waste pollution.

OpenIO-Canada is connected to the Exiobase global MRIO database to model value chains happening outside Canada.

### Getting started

Clone the repository (or download it) and install the different libraries required to run this Python class (requirements.txt).<br>
Note that we recommend working with version **3.9** of Python as we can ensure it works for that specific version.<br>
Go to the doc folder and start with the Running_openIO-Canada.ipynb file to generate the IO tables. You can then explore 
how to use openIO-Canada with the other notebooks.

### Classification detail
The classification used by openIO-Canada is the Input-Output Commodity Classification (IOCC) from StatCan. Unfortunately, 
this classification does not provide a fully detailed structure. It is thus complicated to know where each commodity/service
is actually classified. We recreated this structure based on comparisons with the NAPCS classification. You can find this
structure in the doc/OpenIO-Canada-classification-structure.xlsx file.

### Endogenization of capitals
OpenIO-Canada supports the endogenization of capitals, allowing the user to obtain the tables either with or without the
endgenization. Endogenization consists in integrating the use of capital goods (i.e., any good that is used more than a 
year, such as a building, a machine or a software) in the value chain description of all commodities/services. In other words, 
gross capital formation is typically a final demand and with endogenization it becomes part of the intermediate economy. <br>
So what? After endogenization of capitals there are two main consequences:
1. The emission factors include the impacts of capital goods on the environment. Without endogenization, it's like having
a cradle-to-gate emission factor without the infrastructure. Not performing endogenization of capitals thus results in 
underestimated emission factors.
2. The national/provincial emission estimates are underestimated. While endogenization of capitals does not (or really 
marginally) affect the impact of domestic capital goods, it affects also capital goods from other countries. In other words,
national emission levels, following a consumption approach, without endogenization of capitals includes domestic capital 
goods but excludes foreign capital goods that were used to produce imported commodities/services. Not endogenizing capitals
thus results in an underestimation once again. <br>
Do note however, that endogenization introduces a lot of uncertainties and that studies currently typically do not endogenize capitals. <br>
For more information on endogenization, you can refer to these articles: https://doi.org/10.1021/acs.est.8b02791 https://doi.org/10.1111/jiec.12931

### Basic vs purchaser price
OpenIO-Canada operates at basic price, and so does exiobase, thus ensuring consistency. However, openIO-Canada can still 
provide emission factors at purchaser price for its users. We use average impact of trade and downstream transportation
to estimate these purchaser price emission factors. <br>
What's basic and purchaser price you ask? <br>
The purchaser price is the price that is paid by the final consumer. The basic price is the price of the manufactured 
commodity. These are different, since there are additional costs between the price of the manufactured commodity and the 
final price paid by the consumer. The figure below illustrates the difference in the case of a t-shirt.
<img src="images/prices_explained.png" width="600"/>
While the price of the manufactured t-shirt is 4$, the final consumer pays that t-shirt 10$ because the retailer/wholesaler
makes a profit on the sale, then the t-shirt could be delivered to the consumer and finally taxes are being paid on the t-shirt. <br>
So in the end, if somehow you have access to the breakdown of basic price/retail margins/downstream transportation margins
/taxes you should use the basic price emission factors. In most cases, this breakdown is unavailable. If that is your case, 
then simply use the purchaser price emission factors.

### Contact
maxime.agez@polymtl.ca

### Citation
https://doi.org/10.5281/zenodo.10971810

### Scientific studies using openIO-Canada
- Anne de Bortoli, Maxime Agez, Environmentally-extended input-output analyses efficiently sketch large-scale environmental transition plans: Illustration by Canada's road industry,
Journal of Cleaner Production, Volume 388, 2023, 136039, ISSN 0959-6526, https://doi.org/10.1016/j.jclepro.2023.136039.
- Wambersie, L., & Ouellet-Plamondon, C. (2024). Developing a comprehensive account of embodied emissions within the Canadian construction sector. Journal of Industrial Ecology, 1–14. https://doi.org/10.1111/jiec.13548
- Yoffe, H., et al. (2024). Mapping construction sector greenhouse gas emissions: a crucial step in sustainability meeting increasing housing demands. Environmental research - Infrastructure and sustainability. https://doi.org/10.1088/2634-4505/ad546a