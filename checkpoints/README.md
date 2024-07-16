## Baseline Results

We evaluate the performance of baseline models with and without data augmentation across different domains. While all models show reasonable accuracy under regular conditions, their effectiveness substantially deteriorates when tested in shifted domains such as in the presence of bleeding, low brightness, and smoke. The data augmentations applied, including AutoAugment elastic, and projective transformations, generally failed to improve model performance in these domains. Notably, UNet combined with AutoAugment showed a marginal improvement under smoke conditions but this improvement is inconsistent across all domains.  As shown in the table below, under regular conditions, the models adequately segmented tools from the background, but struggled in areas with appearance change and subtle contrast changes. 

Failures are predominantly under bleeding conditions due to the appearance (color) change of the shaft, leading to incorrect segmentation of the shaft. Under low brightness conditions, the darker the region is, the more likely a failure in segmentation will occur. 

The pervasive false positives under smoke conditions indicate that the models interpret the bright regions (smoke) as the foreground. These results suggest that all baseline models are highly dependent on adequate contrast to successfully distinguish between the tools and the background. They are also sensitive to any appearance change for the tool (bleeding) and background (smoke).

### Quantitative Results 

- DSC

| Architecture \ Domain         | Bleeding            | Smoke               | Low Brightness              |       
|:---------------------:        |:--------:           |:-----:              |:--------------:             |
|       DeepLabv3+              |        0.6896       |      0.6538         |         0.5352              |
|       Segformer               |        0.6802       |      0.6906         |         0.6145              |                                      
|       SETR PUP                |        0.4717       |      0.5848         |         0.4053              |
|       SETR MLA                |        0.5358       |      0.6206         |         0.3385              |                                      
|       SETR Naive              |        0.5064       |      0.6657         |         0.0275              |
|       UNet                    |        0.7052       |      0.6603         |         0.5750              |
|       UNet + AutoAugment      |        0.7910       |      0.8895         |         0.6965              |                                      
|       UNet + Elastic          |        0.6910       |      0.6583         |         0.5190              |                                      
|       UNet + Projective       |        0.6903       |      0.6978         |         0.5792              |                                      
|       SETR MLA + AutoAugment  |        0.1918       |      0.2157         |         0.0934              |                                        
|       SETR MLA + Elastic      |        0.5350       |      0.6192         |         0.3472              |                                      
|       SETR MLA + Projective   |        0.4168       |      0.5766         |         0.1447              |

- NSD

| Architecture \ Domain 	      | Bleeding 	          | Smoke 	            | Low Brightness 	      |
|:---------------------:	      |:--------:	          |:-----:	            |:--------------:	      |
|       DeepLabv3+              |        0.5629       |      0.4637         |         0.4000        |
|       Segformer               |        0.5133       |      0.5266         |         0.4194      	|                                      
|       SETR PUP                |        0.2531       |      0.3354         |         0.2599      	| 
|       SETR MLA                |        0.2798       |      0.3374         |         0.1571      	|                                      
|       SETR Naive              |        0.3312       |      0.4409         |         0.0092     		|  
|       UNet                    |        0.5677       |      0.5084         |         0.4390        |
|       UNet + AutoAugment      |        0.6654       |      0.8152         |         0.5344      	|                                      
|       UNet + Elastic          |        0.5622       |      0.5207         |         0.3931      	|                                      
|       UNet + Projective       |        0.5661       |      0.5702         |         0.4265      	|                                      
|       SETR MLA + AutoAugment  |        0.1172       |      0.0910         |         0.0773      	|                                            
|       SETR MLA + Elastic      |        0.2836       |      0.3363         |         0.1654      	|                                      
|       SETR MLA + Projective   |        0.2240       |      0.3055         |         0.0552      	|
 
### Segmentation Visualizations 
![baseline results](../img/baseline_segmentation_result.png)

