## Baseline Results

We evaluate the performance of baseline models with and without data augmentation across different domains. While all models show reasonable accuracy under regular conditions, their effectiveness substantially deteriorates when tested in shifted domains such as in the presence of bleeding, low brightness, and smoke. The data augmentations applied, including AutoAugment elastic, and projective transformations, generally failed to improve model performance in these domains. Notably, UNet combined with AutoAugment showed a marginal improvement under smoke conditions but this improvement is inconsistent across all domains.  As shown in the table below, under regular conditions, the models adequately segmented tools from the background, but struggled in areas with appearance change and subtle contrast changes. 

Failures are predominantly under bleeding conditions due to the appearance (color) change of the shaft, leading to incorrect segmentation of the shaft. Under low brightness conditions, the darker the region is, the more likely a failure in segmentation will occur. 

The pervasive false positives under smoke conditions indicate that the models interpret the bright regions (smoke) as the foreground. These results suggest that all baseline models are highly dependent on adequate contrast to successfully distinguish between the tools and the background. They are also sensitive to any appearance change for the tool (bleeding) and background (smoke).

### Quantitative Results 

- DSC

| Architecture \ Domain         | Bleeding            | Smoke               | Low Brightness              |checkpoints                |       
|:---------------------:        |:--------:           |:-----:              |:--------------:             |:--------------:           |
|       DeepLabv3+              |        0.6896       |      0.6538         |         0.5352              | [download](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/hding15_jh_edu/EQpXqfvRFDNOgDCXBRKSvrkB_K2x1QTaszFwudz2LH__sQ?e=t5BEQx)|
|       Segformer               |        0.6802       |      0.6906         |         0.6145              |[download](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/hding15_jh_edu/EYKtSiqhRFlNi25NuJorqhsB_DWfQDUi6MfXssawBBkh0A?e=Yp1saj)|                         
|       SETR PUP                |        0.4717       |      0.5848         |         0.4053              |[download](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/hding15_jh_edu/EWKWT3yPxcBLgONbcV2d_bcBF6p3aYj84so-MobPC_1uzQ?e=AuZiMz)|
|       SETR MLA                |        0.5358       |      0.6206         |         0.3385              |[download](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/hding15_jh_edu/EcP1RXjEl_BFn27kVRlzGOsBApYqCVEwiGU9IMEnymlHkg?e=NmHoAo)|                                     
|       SETR Naive              |        0.5064       |      0.6657         |         0.0275              |[download](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/hding15_jh_edu/EYPyKUqjKZZPi03g4MezxnIBTbhdSWG-4xAjFpg8wcTP3A?e=aZUnof)|
|       UNet                    |        0.7052       |      0.6603         |         0.5750              |[download](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/hding15_jh_edu/EQpNqDMnKsBPovg3Cc4pQXUBlYHjZpkpYiNt24vkiQ6BIg?e=hrpQFr)|
|       UNet + AutoAugment      |        0.7910       |      0.8895         |         0.6965              |[download](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/hding15_jh_edu/EQzpFwOV2DZFnW_DoziSXisBypSLo7cN6TUBEbOqcr7DcQ?e=nFDzvD)|                                     
|       UNet + Elastic          |        0.6910       |      0.6583         |         0.5190              |[download](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/hding15_jh_edu/Ec-1J3A5PNNOggjPVvUMl2kBKFcINvYn-biubiAdp0OiSg?e=aUFLZL)|                                    
|       UNet + Projective       |        0.6903       |      0.6978         |         0.5792              |[download](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/hding15_jh_edu/EZquhLnn5MdHpNTD0aUnW_0BCjpEtzXtgmHWG555iQQ3Nw?e=zTqaJo)|                                   
|       SETR MLA + AutoAugment  |        0.1918       |      0.2157         |         0.0934              |[download](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/hding15_jh_edu/ESCzz6JoWzxFjuhYhJiLgzoBtE2c9ZrdS2orM9RDVRwHZw?e=LnKwpn)|                                    
|       SETR MLA + Elastic      |        0.5350       |      0.6192         |         0.3472              |[download](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/hding15_jh_edu/Eamb_Kdiq6FKl46S2NUDdHEBmFJkIrcRRYRidw2IwoWoqw?e=6myjOy)|                                 
|       SETR MLA + Projective   |        0.4168       |      0.5766         |         0.1447              |[download](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/hding15_jh_edu/EUfZvca2_jVFo1h0cLVTmDsBACOQVulbhEQJsCtgmcfyCA?e=lElbF5)|
|       SAM                     |        0.8791       |      0.7681         |         0.8134            |-|
|       MedSAM                  |        0.2213       |      0.1408         |         0.2284            |-|
|       SAM-Med2D               |        0.5302       |      0.5791         |         0.4132            |-|
|       SAM2                    |        0.8628       |      0.8660         |         0.4373            |-|
- NSD

| Architecture \ Domain 	       | Bleeding 	          | Smoke 	             | Low Brightness 	     |
|:---------------------:	       |:--------:	          |:-----:	             |:--------------:	     |
|       DeepLabv3+              |        0.5629       |      0.4637         |         0.4000       |
|       Segformer               |        0.5133       |      0.5266         |         0.4194      	|                                      
|       SETR PUP                |        0.2531       |      0.3354         |         0.2599      	| 
|       SETR MLA                |        0.2798       |      0.3374         |         0.1571      	|                                      
|       SETR Naive              |        0.3312       |      0.4409         |         0.0092     		|  
|       UNet                    |        0.5677       |      0.5084         |         0.4390       |
|       UNet + AutoAugment      |        0.6654       |      0.8152         |         0.5344      	|                                      
|       UNet + Elastic          |        0.5622       |      0.5207         |         0.3931      	|                                      
|       UNet + Projective       |        0.5661       |      0.5702         |         0.4265      	|                                      
|       SETR MLA + AutoAugment  |        0.1172       |      0.0910         |         0.0773      	|                                            
|       SETR MLA + Elastic      |        0.2836       |      0.3363         |         0.1654      	|                                      
|       SETR MLA + Projective   |        0.2240       |      0.3055         |         0.0552      	|
|       SAM                     |        0.5643       |      0.6260         |         0.2281       |
|       MedSAM                  |        0.1636       |      0.1270         |         0.1418       |
|       SAM-Med2D               |        0.3393       |      0.3546         |         0.2423       |
|       SAM2                    |        0.8002       |      0.7808         |         0.2882       |
 
### Segmentation Visualizations 
![baseline results](../img/baseline_segmentation_result.png)
![SAM results](../img/sam_results.png)

