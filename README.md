# Medical Datasets for Machine Learning 机器学习医学数据

## 一.医学影像数据

**1. MedPix**

国家医学图书馆提供了来自13,000名注释患者的53,000张医学图像的MedPix®数据库。需要注册。

Information: [https](/C:%5CUsers%5CQHY%5CAppData%5CRoaming%5CMicrosoft%5CWord%5Chttps):[//medpix.nlm.nih.gov/home](https://medpix.nlm.nih.gov/home)

MedPix®是一个免费开放的在线医学图像、教学案例和临床主题数据库，集成了图像和文本元数据，包括超过12,000个病例场景、9,000个主题和近59,000个图像。

内容材料按疾病部位(器官系统)组织;病理类别;病人资料;通过图像分类和图像标题。可根据患者症状和体征、诊断、器官系统、图像形态和图像描述、关键字、贡献作者和许多其他搜索选项进行搜索。

(1) **A Tour of Unsupervised Deep Learning for Medical Image Analysis**

[Khalid Raza](https://arxiv.org/search/eess?searchtype=author&amp;query=Raza%2C+K), [Nripendra Kumar Singh](https://arxiv.org/search/eess?searchtype=author&amp;query=Singh%2C+N+K) 
[https://arxiv.org/abs/1812.07715](https://arxiv.org/abs/1812.07715)

This review systematically presents various unsupervised models applied to medical image analysis, including autoencoders and its several variants, Restricted Boltzmann machines, Deep belief networks, Deep Boltzmann machine and Generative adversarial network. Future research opportunities and challenges of unsupervised techniques for medical image analysis have also been discussed.

从无监督学习的角度系统介绍了各种医学图像分析模型。

### 2. ABIDE

自闭症脑成像数据交换：发布于2013年，这是一个对自闭症内在大脑结构的大规模评估数据集，包括539名ASD患者和573名正常个体的功能MRI图像。这1112个数据集由结构和静止状态功能MRI数据以及大量的表型信息组成。需要注册。

论文：[http](http://www.ncbi.nlm.nih.gov/pubmed/23774715):[//www.ncbi.nlm.nih.gov/pubmed/23774715](https://blog.csdn.net/indi/abide/)

Information: [http](https://blog.csdn.net/indi/abide/):[//fcon.1000.projects.nitrc.org/indi/abide/
](https://blog.csdn.net/indi/abide/)

(1) **A Novel Transfer Learning Approach to Enhance Deep Neural Network Classification of Brain Functional Connectomes**

Li, Hailong &amp; Parikh, Nehal &amp; He, Lili.（Frontiers in Neuroscience 2018）

Inspired by the transfer learning strategy employed in computer vision, we exploited previously collected resting-state functional MRI data for healthy subjects from existing databases and transferred this knowledge for new disease classification tasks. We developed a deep transfer learning neural network (DTL-NN) framework for enhancing the classification of whole brain functional connectivity patterns.

利用迁移学习在训练样本不足的情况下进行全脑功能连接模式分类

**3. ADNI** **Alzheimer s Disease Neuroimaging Initiative**

关于阿尔茨海默病患者和健康对照的MRI数据库。还有临床，基因组和生物制剂数据。ANDI涉及到的数据集包括如下几部分Clinical Data（临床数据）、MR Image Data（磁共振成像）、Standardized MRI Data Sets、PET Image Data（正电子发射计算机断层扫描）、Gennetic Data（遗传数据）、Biospecimen Data（生物样本数据）。

论文：[http](http://www.neurology.org/content/74/3/201.short):[//www.neurology.org/content/74/3/201.short
](http://www.neurology.org/content/74/3/201.short)
链接：[http](http://adni.loni.usc.edu/data-samples/access-data/):[//adni.loni.usc.edu/data-samples/access-data/](http://adni.loni.usc.edu/data-samples/access-data/)

**(1) Machine learning framework for early MRI-based Alzheimer&#39;s conversion prediction in MCI subjects**

Moradi, Elaheh &amp; Pepe, Antonietta &amp; Gaser, Christian &amp; Huttunen, Heikki &amp; Tohka, Jussi. (NeuroImage 2014)

In this study, we present a novel Magnetic Resonance Imaging (MRI)-based method for predicting the MCI-to-AD conversion from one to three years before the clinical diagnosis. First, we developed a novel MRI biomarker of MCI-to-AD conversion using semi-supervised learning and then integrated it with age and cognitive measures about the subjects using a supervised learning algorithm resulting in what we call the aggregate biomarker.

利用半监督学习实现了阿尔茨海默症的早期检测。

**(2) Machine Learning-Based Method for Personalized and Cost-Effective Detection of Alzheimer&#39;s Disease**

Escudero, Javier &amp; Ifeachor, Emmanuel &amp; Zajicek, John &amp; Green, Colin &amp; Shearer, James &amp; Pearson, Stephen. (IEEE transactions on bio-medical engineering. 2012).

We describe and test a machine learning approach for personalized and cost-effective diagnosis of AD. It uses locally weighted learning to tailor a classifier model to each patient and computes the sequence of biomarkers most informative or cost-effective to diagnose patients.

使用局部加权学习定制个性化分类模型，并计算出最具成本效益的诊断患者的生物标志物序列，用于阿尔茨海默症的早期检测。

### 4. CT Colongraphy for Colon Cancer (Cancer Imaging Archive)

 用于诊断结肠癌的CT扫描。包括没有息肉，6-9mm息肉和超过10 mm息肉的患者的数据。[https](https://wiki.cancerimagingarchive.net/display/Public/CT+COLONOGRAPHY#dc149b9170f54aa29e88f1119e25ba3e):[//wiki.cancerimagingarchive.net/display/Public/CT+COLONOGRAPHY#dc149b9170f54aa29e88f1119e25ba3e](https://wiki.cancerimagingarchive.net/display/Public/CT+COLONOGRAPHY#dc149b9170f54aa29e88f1119e25ba3e)

 **(1)Accurate polyp segmentation for 3D CT colongraphy using multi-staged probabilistic binary learning and compositional model**

Lu, Le &amp; Barbu, Adrian &amp; Wolf, Matthias &amp; Liang, Jianming &amp; Salganicoff, Marcos &amp; Comaniciu, Dorin. (CVPR 2008)

In this paper, we propose a three-staged probabilistic binary classification approach for automatically segmenting polyp voxels from their surrounding tissues in CT. Our system integrates low-, and mid-level information for discriminative learning under local polar coordinates which align on the 3D colon surface around detected polyp.

利用三阶段概率二分类方法对肠息肉组织进行分类。

### 5. Digital Retinal Images for Vessel Extraction (DRIVE)
 DRIVE数据库用于对视网膜图像中的血管分割进行比较研究。它由40张照片组成，其中7张显示轻度早期糖尿病视网膜病变的迹象。

论文：[https](https://ieeexplore.ieee.org/document/1282003)：[//ieeexplore.ieee.org/document/1282003
](https://ieeexplore.ieee.org/document/1282003)
链接：[http](http://www.isi.uu.nl/Research/Databases/DRIVE/download.php)：[//www.isi.uu.nl/Research/Databases/DRIVE/download.php](http://www.isi.uu.nl/Research/Databases/DRIVE/download.php)

**(1) Machine learning identification of diabetic retinopathy from fundus images**

 Gurudath, Nikita and Celenk, Mehmet and Riley, H.

(2014 IEEE Signal Processing in Medicine and Biology Symposium, IEEE SPMB 2014)

https://www.researchgate.net/publication/282931120\_Machine\_learning\_identification\_of\_diabetic\_retinopathy\_from\_fundus\_images

In this research, an approach to automate the identification of the presence of diabetic retinopathy from color fundus images of the retina has been proposed.

利用视网膜眼底图像识别糖尿病视网膜病变。通过高斯滤波实现对输入图像的血管分割，并利用局部熵实现阈值化。

### 6. AMRG Cardiac Atlas

AMRG Cardiac MRI Atlas是奥克兰MRI研究组的西门子Avanto扫描仪获得的正常患者心脏的完整标记MRI图像集。

### **7. Congenital Heart Disease (CHD) Atlas**

先天性心脏病（CHD）Atlas代表成人和患有各种先天性心脏病的儿童的MRI数据集，生理临床数据和计算机模型。

### 8. DETERMINE

 除颤器通过磁共振成像降低风险评估是一项前瞻性，多中心，随机临床试验，用于冠状动脉疾病和轻度至中度左心室功能不全的患者。

### 9. MESA

动脉粥样硬化多族裔研究是一项大规模心血管人群研究（\&gt; 6,500名参与者），在美国的六个中心进行。它的目的是调查亚临床到临床心血管疾病的表现。

1. **Machine Learning Outperforms ACC / AHA CVD Risk Calculator in MESA.**

[https://www.ncbi.nlm.nih.gov/pubmed/30571498](https://www.ncbi.nlm.nih.gov/pubmed/30571498)

 We developed a ML Risk Calculator based on Support Vector Machines ( SVM s) using a 13-year follow up data set from MESA (the Multi-Ethnic Study of Atherosclerosis) of 6459 participants who were atherosclerotic CVD-free at baseline. We provided identical input to both risk calculators and compared their performance. We then used the FLEMENGHO study (the Flemish Study of Environment, Genes and Health Outcomes) to validate the model in an external cohort.

开发了基于支持向量机的风险计算器.

**10. OASIS**

开放获取系列成像研究（OASIS）：OASIS，全称为Open Access Series of Imaging Studies，已经发布了第3代版本，第一次发布于2007年，是一项旨在使科学界免费提供大脑核磁共振数据集的项目。有两个数据集：横截面和纵向集。

(1) 横截面数据集：年轻，中老年，非痴呆和痴呆老年人的横断面MRI数据。该组由416名年龄在18岁至96岁的受试者组成的横截面数据库组成。对于每位受试者，单独获得3或4个单独的T1加权MRI扫描包括扫描会话。受试者都是右撇子，包括男性和女性。100名60岁以上的受试者已经临床诊断为轻度至中度阿尔茨海默病。

(2) 纵向集数据集：非痴呆和痴呆老年人的纵向磁共振成像数据。该集合包括150名年龄在60至96岁的受试者的纵向集合。每位受试者在两次或多次访视中进行扫描，间隔至少一年，总共进行373次成像。对于每个受试者，包括在单次扫描期间获得的3或4次单独的T1加权MRI扫描。受试者都是右撇子，包括男性和女性。在整个研究中，72名受试者被描述为未被证实。包括的受试者中有64人在初次就诊时表现为痴呆症，并在随后的扫描中仍然如此，其中包括51名轻度至中度阿尔茨海默病患者。另外14名受试者在初次就诊时表现为未衰退，随后在随后的访视中表现为痴呆症。

链接：[http](http://www.oasis-brains.org/):[//www.oasis-brains.org/](http://www.oasis-brains.org/)

 **(1)Feature Selection Optimization Using Artificial Immune System Algorithm for Identifying Dementia in MRI Images**

Kakadiaris IA, Vrigkas M, Yen AA, Kuznetsova T, Budoff M, Naghavi M. (J Am Heart Assoc. 2018)

Automatic dementia classification of MRI medical images using machine learning techniques is presented in this paper. For evaluation, MRI images from OASIS dataset are used. MRI images are segmented and features are extracted from segmented image using Discrete Wavelet Transform. Feature selection is via proposed Artificial Immune System (AIS), that searches solution space for correlation based feature selection. Naïve Bayes, CART, C4.5 and K nearest neighbour then classifies the selected features as dementia or non-dementia.

提出了一种基于机器学习技术的MRI医学图像痴呆自动分类方法。

###  11. Isic Archive - Melanoma

这个档案包含23k分类皮肤病变图像。它包含恶性和良性的例子。
每个例子都包含病变的图像，关于病变的元数据（包括分类和分割）和关于患者的元数据。
可以在以下链接中查看数据：[https](https://www.isic-archive.com/)：[//www.isic-archive.com](https://www.isic-archive.com/)

1. **Deep-CLASS at ISIC Machine Learning Challenge 2018**

Sara Nasiri, Matthias Jung, Julien Helsper, Madjid Fathi

Since early 2017, our team has worked on melanoma classification , and has employed deep learning since beginning of 2018 . Deep learning helps researchers absolutely to treat and detect diseases by analyzing medical data (e.g., medical images).

### 12. SCMR Consensus Data

SCMR共识数据集是一组15项混合病理学心脏MRI研究（5项健康，6项心肌梗死，2项心力衰竭和2项肥大），这些研究均来自不同的MR机器（4 GE，5 Siemens，6 Philips），来自不同设备的数据训练出的模型可能不具有迁移性。

### 13. Sunnybrook Cardiac Data

Sunnybrook心脏数据（SCD），也称为2009年心脏MR左心室分割挑战赛数据，包括来自患者和病理混合的45个电影 - MRI图像：健康，肥大，心肌梗塞和心脏衰竭。

链接：[http](http://www.cardiacatlas.org/studies/):[//www.cardiacatlas.org/studies/](http://www.cardiacatlas.org/studies/)

1. **End-to-End Unsupervised Deformable Image Registration with a Convolutional Neural Network**

Bob D. de Vos, Floris F. Berendsen, Max A. Viergever, Marius Staring, Ivana Išgum(DLMIA 2017, ML-CDS 2017)

In this work we propose a deep learning network for deformable image registration (DIRNet). The DIRNet consists of a convolutional neural network (ConvNet) regressor, a spatial transformer, and a resampler. The ConvNet analyzes a pair of fixed and moving images and outputs parameters for the spatial transformer, which generates the displacement vector field that enables the resampler to warp the moving image to the fixed image. The DIRNet is trained end-to-end by unsupervised optimization of a similarity metric between input image pairs.

提出了一个深度学习网络的变形图像注册(DIRNet)。

### 14. Lung Image Database Consortium (LIDC)

**肺部图像数据库联盟** 初步临床研究表明，肺部的螺旋CT扫描可以改善高危人群肺癌的早期检测。图像处理算法有可能有助于螺旋CT研究中的病变检测，并评估连续CT研究中病变大小的稳定性或变化。使用这种计算机辅助算法可以显着提高螺旋CT肺部筛查的灵敏度和特异性，并通过减少解释所需的医生时间来降低成本。

链接：[https](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)：[//wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)

**(1)Predicting LIDC Diagnostic Characteristics by Combining Spatial and Diagnostic Opinions**

Horsthemke, William &amp; Raicu, Daniela &amp; Furst, Jacob. (2010).

This research uses the publicly available Lung Image Database Consortium (LIDC) collection of radiologists, outlines of nodules and ratings of boundary and shape characteristics: spiculation, margin, lobulation, and sphericity. The approach attempts to reduce the observed disagreement between radiologists on the extent of nodules by combining their spatial opinion using probability maps to create regions of interest (ROIs).

感兴趣区(Region of Interest,ROIs) 是图像的一部分，它通过在图像上选择或使用诸如设定阈值(thresholding)等方法生成。

### 15. TCIA Collections

癌症成像数据设置了各种癌症类型（例如癌，肺癌，骨髓瘤）和各种成像方式。癌症成像档案（TCIA）中的图像数据被组织成专门建立的受试者集合。受试者通常具有共同的癌症类型和/或解剖部位（肺，脑等）。下表中的每个链接包含有关集合的科学价值的信息，有关如何获得可用的任何支持非图像数据的信息，以及查看或下载成像数据的链接。为了支持科学研究的可重复性，TCIA支持数字对象标识符（DOI），允许用户共享研究手稿中引用的TCIA数据子集。

链接：[http](http://www.cancerimagingarchive.net/):[//www.cancerimagingarchive.net/](http://www.cancerimagingarchive.net/)

### 16. Belarus tuberculosis portal

白俄罗斯结核病门户:结核病（TB）是白俄罗斯公共卫生的一个主要问题。最近，MDR / XDR TB和HIV / TB的出现和发展需要长期治疗。许多和最严重的病例通常在全国各地传播到不同的结核病诊所。通过使用包含患者放射图像，实验室工作和临床数据的通用数据库，将大大提高领导白俄罗斯结核病专家跟踪此类患者的能力。这也将显着提高对治疗方案的依从性，从而更好地记录治疗结果。在门户数据库中纳入临床病例的标准 - 被诊断或怀疑患有耐多药结核病的肺病和结核病RSPC的耐多药结核病科住院患者，

链接：[http](http://tuberculosis.by/)：[//tuberculosis.by/](http://tuberculosis.by/)

**(1) Methods for Genome-Wide Analysis of MDR and XDR Tuberculosis from Belarus**

Sergeev R., Kavaliou I., Gabrielian A., Rosenthal A., Tuzikov A. (ISBRA 2016)

Emergence of drug-resistant microorganisms has been recognized as a serious threat to public health since the era of chemotherapy began. This problem is extensively discussed in the context of tuberculosis treatment. Alterations in pathogen genomes are among the main mechanisms by which microorganisms exhibit drug resistance. Analysis of the reported cases and discovery of new resistance-associated mutations may contribute greatly to the development of new drugs and effective therapy management. The proposed methodology allows identifying genetic changes and assessing their contribution to resistance phenotypes.

通过基因组分析识别遗传变化并评估其对耐药表型的贡献。

### 17. DDSM：Digital Database for Screening Mammography

用于筛查乳腺摄影的数字数据库:用于筛查乳房摄影的数字数据库（DDSM），是乳房摄影图像分析研究界使用的资源。该项目的主要支持是美国陆军医学研究和装备司令部乳腺癌研究计划的资助。DDSM项目是由马萨诸塞州综合医院（D. Kopans，R. Moore），南佛罗里达大学（K. Bowyer）和桑迪亚国家实验室（P. Kegelmeyer）共同参与的合作项目。该数据库的主要目的是促进计算机算法开发中的合理研究，以帮助筛选。数据库的次要目的可能包括开发算法以帮助诊断和开发教学或培训辅助工具。该数据库包含大约2,500项研究。每项研究包括每个乳房的两个图像，以及一些相关的患者信息（研究时的年龄，ACR乳房密度评级，异常的微妙评级，异常的ACR关键字描述）和图像信息（扫描仪，空间分辨率）。包含可疑区域的图像具有关于可疑区域的位置和类型的像素级&quot;基础事实&quot;信息。

链接：[http://marathon.csee.usf.edu/Mammography/Database.html](http://marathon.csee.usf.edu/Mammography/Database.html)

**(1)AdaBoost-based multiple SVM-RFE for classification of mammograms in DDSM**

Yoon, Sejong &amp; Kim, Saejoon. (BMC medical informatics and decision making 2009).

We propose a feature selection method based on multiple support vector machine recursive feature elimination (MSVM-RFE). We compared our method with four previously proposed feature selection methods which use support vector machine as the base classifier. Experiments were performed on lesions extracted from the Digital Database of Screening Mammography, the largest public digital mammography database available.

提出了一种基于多支持向量机递归特征消除(MSVM-RFE)的特征选择方法，并在DDSM数据集实验。

### 18. INbreast: Database for Digital Mammography

数字乳腺摄影数据库:INbreast数据库是一个乳腺摄影数据库，其图像位于大学医院（医院deSãoJoão，乳房中心，葡萄牙波尔图）的乳房中心。INbreast共有115例（410张图像），其中90例来自两个乳房的女性（每例4张），25例来自乳房切除术患者（每例2张）。包括几种类型的病变（肿块，钙化，不对称和扭曲）。专家制作的精确轮廓也以XML格式提供。

[http://medicalresearch.inescporto.pt/breastresearch/index.php/Get\_INbreast\_Database](http://medicalresearch.inescporto.pt/breastresearch/index.php/Get_INbreast_Database)

 **(1)Identification of Individualized Feature Combinations for Survival Prediction in Breast Cancer: A Comparison of Machine Learning Techniques**

Vanneschi, Leonardo &amp; Farinaccio, Antonella &amp; Giacobini, Mario &amp; Mauri, Giancarlo &amp; Antoniotti, Marco &amp; Provero, Paolo. (2010).

Here we investigate the use of several machine learning techniques to classify breast cancer patients using one of such signatures, the well established 70-gene signature. We show that Genetic Programming performs significantly better than Support Vector Machines, Multilayered Perceptron and Random Forest in classifying patients from the NKI breast cancer dataset, and slightly better than the scoring-based method originally proposed by the authors of the seventy-gene signature. Furthermore, Genetic Programming is able to perform an automatic feature selection. Since the performance of Genetic Programming is likely to be improvable compared to the out-of-the-box approach used here, and given the biological insight potentially provided by the Genetic Programming solutions, we conclude that Genetic Programming methods are worth further investigation as a tool for cancer patient classification based on gene expression data.

### 19. mini-MIAS：MIAS MiniMammographic Database

MIAS全称为MiniMammographic Database，是乳腺图像数据库。乳房X线摄影图像分析协会（MIAS）是一个英国研究小组的组织，有兴趣了解乳房X线照片并生成数字乳房X线照片数据库。采用英国国家乳房筛查计划的胶片已经数字化为50微米像素边缘，使用Joyce-Loebl扫描微密度计，光学密度范围为0-3.2的线性装置，并用8位字表示每个像素。该数据库包含322个数字化电影，可在2.3GB 8mm（ExaByte）磁带上使用。它还包括放射科医师对可能存在的任何异常位置的&quot;真相&quot;标记。数据库已减少到200微米像素边缘并填充/剪裁，以便所有图像都是1024x1024。

链接：[http](http://peipa.essex.ac.uk/info/mias.html):[//peipa.essex.ac.uk/info/mias.html](http://peipa.essex.ac.uk/info/mias.html)

数据集地址：

http://peipa.essex.ac.uk/pix/mias/all-mias.tar.gz

[https://www.repository.cam.ac.uk/handle/1810/250394?show=full](https://www.repository.cam.ac.uk/handle/1810/250394?show=full)

乳腺MG数据（Breast Mammography）有个专门的database，可以查看很多数据集，链接地址：http://www.mammoimage.org/databases/

 **(1)Assessment of a novel mass detection algorithm in mammograms**

Kozegar, Ehsan &amp; Soryani, Mohsen &amp; Minaei, Behrouz &amp; Domingues, Ines. (Journal of cancer research and therapeutics 2013).

Context: Mammography is the most effective procedure for an early detection of the breast abnormalities. Masses are a type of abnormality, which are very difficult to be visually detected on mammograms.

Aims: In this paper an efficient method for detection of masses in mammograms is implemented. Settings and Design: The proposed mass detector consists of two major steps. In the first step, several suspicious regions are extracted from the mammograms using an adaptive thresholding technique. In the second step, false positives originating by the previous stage are reduced by a machine learning approach.

Materials and Methods: All modules of the mass detector were assessed on mini-MIAS database. In addition, the algorithm was tested on INBreast database for more validation. Results: According to FROC analysis, our mass detection algorithm outperforms other competing methods. Conclusions: We should not just insist on sensitivity in the segmentation phase because if we forgot FP rate, and our goal was just higher sensitivity, then the learning algorithm would be biased more toward false positives and the sensitivity would decrease dramatically in the false positive reduction phase. Therefore, we should consider the mass detection problem as a cost sensitive problem because misclassification costs are not the same in this type of problems.

### 20.Prostate

前列腺:据报道，前列腺癌（CaP）在全球范围内是第二常见的男性癌症，占13.6％（Ferlay等人（2010））。据统计，在2008年，新诊断病例的数量估计为899,000例，其中不少于258例，100例死亡（Ferlay等人（2010年））。

磁共振成像（MRI）提供允许诊断和定位CaP的成像技术。I2CVB提供多参数MRI数据集，以帮助开发计算机辅助检测和诊断（CAD）系统。访问：[http](http://i2cvb.github.io/):[//i2cvb.github.io/](http://i2cvb.github.io/)

链接：[http](http://www.medinfo.cs.ucy.ac.cy/index.php/facilities/32-software/218-datasets):[//www.medinfo.cs.ucy.ac.cy/index.php/facilities/32-software/218-datasets](http://www.medinfo.cs.ucy.ac.cy/index.php/facilities/32-software/218-datasets)

**(1) Machine Learning for Survival Analysis: A Case Study on Recurrence of Prostate Cancer.**

Zupan, Blaz &amp; Demšar, Janez &amp; Kattan, Michael &amp; Beck, J.. (Artificial intelligence in medicine. 2000).

In this paper we propose a schema that enables the use of classification methods--including machine learning classifiers--for survival analysis. To appropriately consider the follow-up time and censoring, we propose a technique that, for the patients for which the event did not occur and have short follow-up times, estimates their probability of event and assigns them a distribution of outcome accordingly. Since most machine learning techniques do not deal with outcome distributions, the schema is implemented using weighted examples. To show the utility of the proposed technique, we investigate a particular problem of building prognostic models for prostate cancer recurrence, where the sole prediction of the probability of event (and not its probability dependency on time) is of interest.

### 21. DICOM image sample sets

链接：[http](http://www.osirix-viewer.com/resources/dicom-image-library/)：[//www.osirix-viewer.com/resources/dicom-image-library/](http://www.osirix-viewer.com/resources/dicom-image-library/)

DICOM（Digital Imaging and Communications in Medicine）即医学数字成像和通信，是医学图像和相关信息的国际标准（ISO 12052）。它定义了质量能满足临床需要的可用于数据交换的医学图像格式。

### 22. SCR database: Segmentation in Chest Radiographs

胸部X光片中的分割:胸片中解剖结构的自动分割对于这些图像中的计算机辅助诊断非常重要。已经建立了SCR数据库，以便于对标准后胸部前胸片中肺野，心脏和锁骨的分割进行比较研究。

链接：[http](http://www.isi.uu.nl/Research/Databases/SCR/)：[//www.isi.uu.nl/Research/Databases/SCR/](http://www.isi.uu.nl/Research/Databases/SCR/)

**(1)Discriminative learning of deformable contour models**

Boussaid, Haithem &amp; Kokkinos, Iasonas &amp; Paragios, Nikos. (IEEE International Symposium on Biomedical Imaging. 2014).

In this work we propose a machine learning approach to improve shape detection accuracy inmedical images with deformable contour models (DCMs). Our DCMs can efficiently recover globally optimal solutions that take into account constraints on shape and appearance in the model fitting criterion; our model can also deal with global scale variations by operating in a multi-scale pyramid. Our main contribution consists in formulating the task of learning the DCM score function as a large-margin structured prediction problem. Our algorithm trains DCMs in an joint manner - all the parameters are learned simultaneously, while we use rich local features for landmark localization.

### 23. Medical Image Databases &amp; Libraries

医学图像数据库和图书馆

链接：[http](http://www.omnimedicalsearch.com/image_databases.html)：[//www.omnimedicalsearch.com/image\_databases.html](http://www.omnimedicalsearch.com/image_databases.html)

### 24. VIA Group Public Databases

文件化的图像数据库对于定量图像分析工具的开发至关重要，特别是对于计算机辅助诊断(CAD)任务。与I-ELCAP小组合作，我们建立了两个公共图像数据库，其中包含DICOM格式的肺CT图像以及放射科医生的异常记录。

链接：[http](http://www.via.cornell.edu/databases/):[//www.via.cornell.edu/databases/](http://www.via.cornell.edu/databases/)

**25. CVonline**

对计算机视觉研究和算法评估有用的图像和视频数据库的整理列表。

链接：[http://homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm](http://homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm)

生物/医学部分

[http://homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm#biomed](http://homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm#biomed)

1. [2008 MICCAI MS Lesion Segmentation Challenge](https://www.nitrc.org/projects/msseg) (National Institutes of Health Blueprint for Neuroscience Research) [Before 28/12/19] 神经科学研究
2. [ASU DR-AutoCC Data](https://github.com/ragavvenkatesan/np-mil/blob/master/data/DR_data.mat) - a Multiple-Instance Learning feature space for a diabetic retinopathy classification dataset (Ragav Venkatesan, Parag Chandakkar, Baoxin Li - Arizona State University) [Before 28/12/19] 糖尿病性视网膜病变分类数据集
3. Aberystwyth Leaf Evaluation Dataset - Timelapse plant images with hand marked up leaf-level segmentations for some time steps, and biological data from plant sacrifice. (Bell, Jonathan; Dee, Hannah M.) [Before 28/12/19]
4. [ADP: Atlas of Digital Pathology](http://www.dsp.utoronto.ca/projects/ADP/ADP_Database/) - 17,668 histological patch images extracted from 100 slides annotated with up to 57 hierarchical tissue types (HTTs) from different organs - the aim is to provide training data for supervised multi-label learning of tissue types in a digitized whole slide image (Hosseini, Chan, Tse, Tang, Deng, Norouzi, Rowsell, Plataniotis, Damaskinos) [14/1/20] 组织学斑块图像
5. [Annotated Spine CT Database](http://spineweb.digitalimaginggroup.ca/spineweb/index.php?n=Main.Datasets) for Benchmarking of Vertebrae Localization, 125 patients, 242 scans (Ben Glockern) [Before 28/12/19] 脊柱CT数据库
6. [BRATS](http://braintumorsegmentation.org/) - the identification and segmentation of tumor structures in multiparametric magnetic resonance images of the brain (TU Munchen etc.) [Before 28/12/19]
7. [Breast Ultrasound Dataset B](http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php) - 2D Breast Ultrasound Images with 53 malignant lesions and 110 benign lesions. (UDIAT Diagnostic Centre, M.H. Yap, R. Marti) [Before 28/12/19] 乳房超声图像
8. [Calgary-Campinas Public Brain MR Dataset](https://sites.google.com/view/calgary-campinas-dataset/home): T1-weighted brain MRI volumes acquired in 359 subjects on scanners from three different vendors (GE, Philips, and Siemens) and at two magnetic field strengths (1.5 T and 3 T). The scans correspond to older adult subjects. (Souza, Roberto, Oeslle Lucena, Julia Garrafa, David Gobbi, Marina Saluzzi, Simone Appenzeller, Leticia Rittner, Richard Frayne, and Roberto Lotufo) [Before 28/12/19] 脑MR数据集
9. [CAMEL colorectal adenoma dataset](https://github.com/ThoroughImages/CAMEL) - image-level labels for weakly supervised learning containing 177 whole slide images (156 contain adenoma) gathered and labeled by pathologists (Song and Wang) [29/12/19] CAMEL大肠腺瘤数据集
10. [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) - a large dataset of chest X-rays and competition for automated chest x-ray interpretation, which features uncertainty labels and radiologist-labeled reference standard evaluation sets (Irvin, Rajpurkar et al) [Before 28/12/19] 胸部X射线数据集
11. [Cholec80](http://camma.u-strasbg.fr/datasets): 80 gallbladder laparoscopic videos annotated with phase and tool information. (Andru Putra Twinanda) [Before 28/12/19] 胆囊腹腔镜视频
12. [CRCHistoPhenotypes - Labeled Cell Nuclei Data](http://www.warwick.ac.uk/BIAlab/data/CRChistoLabeledNucleiHE/) - colorectal cancer?histology images?consisting of nearly 30,000 dotted nuclei with over 22,000 labeled with the cell type (Rajpoot + Sirinukunwattana) [Before 28/12/19] 大肠癌的组织学图像
13. [Cavy Action Dataset](http://www.inf-cv.uni-jena.de/Research/Datasets/Cavy+Dataset.html) - 16 sequences with 640 x 480 resolutions recorded at 7.5 frames per second (fps) with approximately 31621506 frames in total (272 GB) of interacting cavies (guinea pig) (Al-Raziqi and Denzler) [Before 28/12/19] 豚鼠行动数据集
14. [Cell Tracking Challenge Datasets](http://www.codesolorzano.com/celltrackingchallenge/Cell_Tracking_Challenge/Datasets.html) - 2D/3D time-lapse video sequences with ground truth(Ma et al., Bioinformatics 30:1609-1617, 2014) [Before 28/12/19]
15. [Computed Tomography Emphysema Database](http://image.diku.dk/emphysema_database/) (Lauge Sorensen) [Before 28/12/19] 肺气肿数据库
16. [COPD Machine Learning Dataset](http://bigr.nl/research/projects/copd) - A collection of feature datasets derived from lung computed tomography (CT) images, which can be used in diagnosis of chronic obstructive pulmonary disease (COPD). The images in this database are weakly labeled, i.e. per image, a diagnosis(COPD or no COPD) is given, but it is not known which parts of the lungs are affected. Furthermore, the images were acquired at different sites and with different scanners. These problems are related to two learning scenarios in machine learning, namely multiple instance learning or weakly supervised learning, and transfer learning or domain adaptation. (Veronika Cheplygina, Isabel Pino Pena, Jesper Holst Pedersen, David A. Lynch, Lauge S., Marleen de Bruijne) [Before 28/12/19] 肺部计算机断层扫描（CT）图像的特征数据集，可用于诊断慢性阻塞性肺疾病（COPD
17. [CREMI: MICCAI 2016 Challenge](https://cremi.org/data) - 6 volumes of electron microscopy of neural tissue,neuron and synapse segmentation, synaptic partner annotation. (Jan Funke, Stephan Saalfeld, Srini Turaga, Davi Bock, Eric Perlman) [Before 28/12/19] 神经组织电子显微镜
18. [CRIM13 Caltech Resident-Intruder Mouse dataset](http://www.vision.caltech.edu/Video_Datasets/CRIM13/CRIM13/Main.html) - 237 10 minute videos (25 fps) annotated with actions (13 classes) (Burgos-Artizzu, Dollar, Lin, Anderson and Perona) [Before 28/12/19]
19. [CVC colon DB](http://mv.cvc.uab.es/projects/colon-qa/cvccolondb) - annotated video sequences of colonoscopy video. It contains 15 short colonoscopy sequences, coming from 15 different studies. In each sequence one polyp is shown. (Bernal, Sanchez, Vilarino) [Before 28/12/19] 结肠数据库
20. [DIADEM: Digital Reconstruction of Axonal and Dendritic Morphology Competition](http://diademchallenge.org/) (Allen Institute for Brain Science et al) [Before 28/12/19] 轴突和树突形态竞赛的数字重建
21. [DIARETDB1 - Standard Diabetic Retinopathy Database](http://www2.it.lut.fi/project/imageret/diaretdb1/) (Lappeenranta Univ of Technology) [Before 28/12/19] DIARETDB1-标准糖尿病性视网膜病数据库
22. [DRIVE: Digital Retinal Images for Vessel Extraction](http://www.isi.uu.nl/Research/Databases/DRIVE/) (Univ of Utrecht) [Before 28/12/19] 用于提取血管的数字视网膜图像
23. [DeformIt 2.0](http://www.cs.sfu.ca/~hamarneh/software/DeformIt/index.html) - Image Data Augmentation Tool: Simulate novel images with ground truth segmentations from a single image-segmentation pair (Brian Booth and Ghassan Hamarneh) [Before 28/12/19]
24. [Deformable Image Registration Lab dataset](https://www.dir-lab.com/) - for objective and rigrorous evaluation of deformable image registration (DIR) spatial accuracy performance. (Richard Castillo et al.) [Before 28/12/19]
25. [DERMOFIT Skin Cancer Dataset](http://homepages.inf.ed.ac.uk/rbf/DERMOFIT/datasets.htm) - 1300 lesions from 10 classes captured under identical controlled conditions. Lesion segmentation masks are included (Fisher, Rees, Aldridge, Ballerini, et al) [Before 28/12/19] [皮肤癌数据集](http://homepages.inf.ed.ac.uk/rbf/DERMOFIT/datasets.htm)
26. [Dermoscopy images](http://dermoscopic.blogspot.com/) (Eric Ehrsam) [Before 28/12/19]
27. [EATMINT (Emotional Awareness Tools for Mediated INTeraction) database](https://eatmint.unige.ch/home.php) - The EATMINT database contains multi-modal and multi-user recordings of affect and social behaviors in a collaborative setting. (Guillaume Chanel, Gaelle Molinari, Thierry Pun, Mireille Betrancourt) [Before 28/12/19]
28. [EPT29.](http://web.engr.oregonstate.edu/~tgd/bugid/ept29/)This database contains 4842 images of 1613 specimens of 29 taxa of EPTs:(Tom etc.) [Before 28/12/19]
29. [EyePACS](http://www.eyepacs.com/data-analysis) - retinal image database is comprised of over 3 million retinal images of diverse populations with various degrees of diabetic retinopathy (EyePACS) [Before 28/12/19]
30. [FIRE Fundus Image Registration Dataset](http://www.ics.forth.gr/cvrl/fire) - 134 retinal image pairs and groud truth for registration. (FORTH-ICS) [Before 28/12/19] 眼底图像配准数据集
31. [FMD - Fluorescence Microscopy Denoising dataset](https://drive.google.com/drive/folders/1aygMzSDdoq63IqSk-ly8cMq0_owup8UM) - 12,000 real fluorescence microscopy images (Zhang, Zhu, Nichols, Wang, Zhang, Smith, Howard) [Before 28/12/19]
32. [FocusPath](https://github.com/mahdihosseini/FoucsPath) - Focus Quality Assessment for Digital Pathology (Microscopy) Images. 864 image pathes are naturally blurred by 16 levels of out-of-focus lens provided with GT scores of focus levels. (Hosseini, Zhang, Plataniotis) [Before 28/12/19]
33. [Histology Image Collection Library (HICL)](http://medisp.bme.teiath.gr/hicl/index.html) - The HICL is a compilation of 3870histopathological images (so far) from various diseases, such as brain cancer,breast cancer and HPV (Human Papilloma Virus)-Cervical cancer. (Medical Image and Signal Processing (MEDISP) Lab., Department of BiomedicalEngineering, School of Engineering, University of West Attica) [Before 28/12/19] 组织病理学图像的汇编
34. [Honeybee segmentation dataset](https://groups.oist.jp/bptu/honeybee-tracking-dataset) - It is a dataset containing positions and orientation angles of hundreds of bees on a 2D surface of honey comb. (Bozek K, Hebert L, Mikheyev AS, Stephesn GJ) [Before 28/12/19]
35. [IIT MBADA mice](https://www.iit.it/research/lines/pattern-analysis-and-computer-vision/pavis-datasets/531-mice-behaviour-analysis) - Mice behavioral data. FLIR A315, spacial resolution of 320??240px at 30fps, 50x50cm open arena, two experts for three different mice pairs, mice identities. (Italian Inst. of Technology, PAVIS lab) [Before 28/12/19]
36. [Indian Diabetic Retinopathy Image Dataset](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) - This dataset consists of retinal fundus images annotated at pixel-level for lesions associated with Diabetic Retinopathy. Also, it provides the disease severity of diabetic retinopathy and diabetic macular edema. This dataset is useful for development and evaluation of image analysis algorithms for early detection of diabetic retinopathy. (Prasanna Porwal, Samiksha Pachade, Ravi Kamble, Manesh Kokare, Girish Deshmukh, Vivek Sahasrabuddhe, Fabrice Meriaudeau) [Before 28/12/19] 印度糖尿病性视网膜病变图像数据集
37. [IRMA(Image retrieval in medical applications)](https://ganymed.imib.rwth-aachen.de/irma/datasets_en.php?SELECTED=00009) - This collection compiles anonymous radiographs (Deserno TM, Ott B) [Before 28/12/19]
38. [IVDM3Seg](https://ivdm3seg.weebly.com/data.html) - 24 3D multi-modality MRI data sets of at least 7 IVDs of the lower spine, collected from 12 subjects in two different stages (Zheng, Li, Belavy) [Before 28/12/19]
39. [JIGSAWS](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/) - JHU-ISI Surgical Gesture and Skill Assessment Working Set (a surgical activity dataset for human motion modeling, captured using the da Vinci Surgical System from eight surgeons with different levels of skill performing five repetitions of three elementary surgical tasks. It contains: kinematic and video data, plus manual annotations. (Carol Reiley and Balazs Vagvolgyi) [Before 28/12/19]
40. [KID](http://is-innovation.eu/kid) - A capsule endoscopy database for medical decision support (Anastasios Koulaouzidis and Dimitris Iakovidis) [Before 28/12/19]
41. [Leaf Segmentation Challenge](http://www.plant-phenotyping.org/CVPPP2014-dataset)Tobacco and arabidopsis plant images (Hanno Scharr, Massimo Minervini, Andreas Fischbach, Sotirios A. Tsaftaris) [Before 28/12/19]
42. [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) - Lung Image Database Consortium image collection (LIDC-IDRI) consists of diagnostic and lung cancer screening thoracic computed tomography (CT) scans with marked-up annotated lesions. (Before 30/12/19) [Before 28/12/19]
43. [LITS Liver Tumor Segmentation](http://www.lits-challenge.com/) - 130 3D CT scans with segmentations of the liver and liver tumor. Public benchmark with leaderboard at Codalab.org (Patrick Christ) [Before 28/12/19]
44. [Mammographic Image Analysis Homepage](http://www.mammoimage.org/databases/) - a collection of databases links [Before 28/12/19]
45. [Medical image database](http://onlinemedicalimages.com/) - Database of ultrasound images of breast abnormalities with the ground truth. (Prof. Stanislav Makhanov, biomedsiit.com) [Before 28/12/19]
46. [MiniMammographic Database](http://www.mammoimage.org/databases/) (Mammographic Image Analysis Society) [Before 28/12/19]
47. [MIT CBCL Automated Mouse Behavior Recognition datasets](http://cbcl.mit.edu/software-datasets/hueihan/) (Nicholas Edelman) [Before 28/12/19]
48. [Moth fine-grained recognition](http://www.inf-cv.uni-jena.de/fgvcbiodiv) - 675 similar classes, 5344 images (Erik Rodner et al) [Before 28/12/19]
49. [Mouse Embryo Tracking Database](http://celltracking.bio.nyu.edu/) - cell division event detection (Marcelo Cicconet, Kris Gunsalus) [Before 28/12/19]
50. [MUCIC: Masaryk University Cell Image Collection](http://cbia.fi.muni.cz/datasets/) - 2D/3D synthetic images of cells/tissues for benchmarking(Masaryk University) [Before 28/12/19]
51. [NIH Chest X-ray Dataset](https://www.kaggle.com/nih-chest-xrays/data) - 112,120 X-ray images with disease labels from 30,805 unique patients. (NIH) [Before 28/12/19]
52. [OASIS](http://www.oasis-brains.org/) - Open Access Series of Imaging Studies - 500+ MRI data sets of the brain (Washington University, Harvard University, Biomedical Informatics Research Network) [Before 28/12/19]
53. [Plant Phenotyping Datasets](http://www.plant-phenotyping.org/datasets-home) - plant data suitable for plant and leaf detection, segmentation, tracking, and species recognition (M. Minervini, A. Fischbach, H. Scharr, S. A. Tsaftaris) [Before 28/12/19]
54. [RatSI: Rat Social Interaction Dataset](http://www.noldus.com/innovationworks/datasets/ratsi) - 9 fully annotated (11 class) videos (15 minute, 25 FPS) of two rats interacting socially in a cage (Malte Lorbach, Noldus Information Technology) [Before 28/12/19]
55. [Retinal fundus images - Ground truth of vascular bifurcations and crossovers](http://www.cs.rug.nl/~imaging/databases/retina_database/retinalfeatures_database.html) (Univ of Groningen) [Before 28/12/19]
56. [SCORHE](https://scorhe.nih.gov/) - 1, 2 and 3 mouse behavior videos, 9 behaviors, (Ghadi H. Salem, et al, NIH) [Before 28/12/19]
57. [SLP (Simultaneously-collected multimodal Lying Pose)](https://web.northeastern.edu/ostadabbas/2019/06/27/multimodal-in-bed-pose-estimation/) - large scale dataset on in-bed poses includes: 2 Data Collection Settings: (a) Hospital setting: 7 participants, and (b) Home setting: 102 participants (29 females, age range: 20-40). 4 Imaging Modalities: RGB (regular webcam), IR (FLIR LWIR camera), DEPTH (Kinect v2) and Pressure Map (Tekscan Pressure Sensing Map). 3 Cover Conditions: uncover, bed sheet, and blanket. Fully labeled poses with 14 joints. (Ostadabbas and Liu) [2/1/20]
58. [SNEMI3D](http://brainiac2.mit.edu/SNEMI3D/) - 3D Segmentation of neurites in EM images [Before 28/12/19]
59. [STructured Analysis of the Retina](http://cecas.clemson.edu/~ahoover/stare/) - DESCRIPTION(400+ retinal images, with ground truth segmentations and medical annotations) (Before 30/12/19) [Before 28/12/19] 视网膜结构分析
60. [Spine and Cardiac data](http://www.digitalimaginggroup.ca/members/shuo.php) (Digital Imaging Group of London Ontario, Shuo Li) [Before 28/12/19] 脊柱和心脏数据
61. [Stonefly9](http://web.engr.oregonstate.edu/~tgd/bugid/stonefly9/)This database contains 3826 images of 773 specimens of 9 taxa of Stoneflies (Tom etc.) [Before 28/12/19]
62. [Synthetic Migrating Cells](http://www.phagosight.org/synData.php) -Six artificial migrating cells (neutrophils) over 98 time frames, various levels of Gaussian/Poisson noise and different paths characteristics with ground truth. (Dr Constantino Carlos Reyes-Aldasoro et al.) [Before 28/12/19]
63. [UBFC-RPPG Dataset](https://sites.google.com/view/ybenezeth/ubfcrppg) - remote photoplethysmography (rPPG) video data and ground truth acquired with a CMS50E transmissive pulse oximeter (Bobbia, Macwan, Benezeth, Mansouri, Dubois) [Before 28/12/19]
64. [Uni Bremen Open, Abdominal Surgery RGB Dataset](http://cgvr.informatik.uni-bremen.de/research/asula/index.shtml) - Recording of a complete, open, abdominal surgery using a Kinect v2 that was mounted directly above the patient looking down at patient and staff. (Joern Teuber, Gabriel Zachmann, University of Bremen) [Before 28/12/19]
65. [Univ of Central Florida - DDSM: Digital Database for Screening Mammography](http://marathon.csee.usf.edu/Mammography/Database.html) (Univ of Central Florida) [Before 28/12/19]
66. [VascuSynth](http://vascusynth.cs.sfu.ca/) - 120 3D vascular tree like structures with ground truth (Mengliu Zhao, Ghassan Hamarneh) [Before 28/12/19]
67. [VascuSynth](http://vascusynth.cs.sfu.ca/Data.html) - Vascular Synthesizer generates vascular trees in 3D volumes. (Ghassan Hamarneh, Preet Jassi, Mengliu Zhao) [Before 28/12/19]
68. [York Cardiac MRI dataset](http://www.cse.yorku.ca/~mridataset/) (Alexander Andreopoulos) [Before 28/12/19]

### 26. USC-SIPI

图像数据库 USC-SIPI图像数据库是数字化图像的集合。它主要用于支持图像处理，图像分析和机器视觉方面的研究。USC-SIPI图像数据库的第一版于1977年发布，此后又添加了许多新图像。

数据库根据图片的基本特征分为卷。每个体积中的图像具有各种尺寸，例如256×256像素，512×512像素或1024×1024像素。所有图像对于黑白图像是8位/像素，对于彩色图像是24位/像素。目前提供以下卷：

Textures         Brodatz textures, texture mosaics, etc.

Aerials         High altitude aerial images

Miscellaneous         Lena, the mandrill, and other favorites

Sequences         Moving head, fly-overs, moving vehicles

链接：[http://sipi.usc.edu/database/](http://sipi.usc.edu/database/)

### 27. Histology (CIMA) dataset

组织学数据集：不同染色切片的图像配准

该数据集由2D组织学显微镜组织切片组成，用不同的染色剂染色，并且标志物表示每个切片中的关键点。任务是图像配准 - 将特定图像集（连续污点切割）中的所有切片对齐在一起，例如对准初始图像平面。这些图像的主要挑战如下：非常大的图像尺寸，外观差异以及缺乏独特的外观对象。该数据集包含108个图像对和手动放置的标记，用于登记质量评估。

链接: [http://cmp.felk.cvut.cz/~borovji3/?page=dataset](http://cmp.felk.cvut.cz/~borovji3/?page=dataset)

**(1)Machine learning approach for segmenting glands in colon histology images using local intensity and texture features**

Khatun, Rupali, and Soumick Chatterjee. (IACC 2018)

The principal objective of this project is to assist the pathologist to accurate detection of colon cancer. In this paper, the authors have proposed an algorithm for an automatic segmentation of glands in colon histology using local intensity and texture features. Here the dataset images are cropped into patches with different window sizes and taken the intensity of those patches, and also calculated texture-based features. Random forest classifier has been used to classify this patch into different labels. A multilevel random forest technique in a hierarchical way is proposed. This solution is fast, accurate and it is very much applicable in a clinical setup.

### 28. ChestX-ray14

ChestX-ray14 是由NIH研究院提供的，其中包含了30,805名患者的112,120个单独标注的14种不同肺部疾病（肺不张、变实、浸润、气胸、水肿、肺气肿、纤维变性、积液、肺炎、胸膜增厚、心脏肥大、结节、肿块和疝气）的正面胸部 X 光片。研究人员对数据采用NLP方法对图像进行标注。利用深度学习的技术早期发现并识别胸透照片中肺炎等疾病对增加患者恢复和生存的最佳机会至关重要。

数据集地址：

https://www.kaggle.com/nih-chest-xrays/data

https://nihcc.app.box.com/v/ChestXray-NIHCC

**(1)CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning**

Pranav Rajpurkar\*, Jeremy Irvin\*, Kaylie Zhu, Brandon Yang, Hershel Mehta, Tony Duan, Daisy Ding, Aarti Bagul, Curtis Langlotz, Katie Shpanskaya, Matthew P. Lungren, Andrew Y. Ng

提出121层的卷积神经网络CheXNet用于肺炎检测, 超过执业放射科医生的水平

### 29. NSCLC

发布于2018年，来自斯坦福大学。数据集来自211名受试者的非小细胞肺癌（NSCLC）队列的独特放射基因组数据集。该数据集包括计算机断层扫描（CT），正电子发射断层扫描（PET）/ CT图像。创建该数据集是为了便于发现基因组和医学图像特征之间的基础关系，以及预测医学图像生物标记的开发和评估。

数据集地址：

https://wiki.cancerimagingarchive.net/display/Public/NSCLC+Radiogenomics

### 30. DeepLesion

DeepLesion由美国国立卫生研究院临床中心（NIHCC）的团队开发，是迄今规模最大的多类别、病灶级别标注临床医疗CT图像开放数据集。在该数据库中图像包括多种病变类型，目前包括4427个患者的32,735 张CT图像及病变信息，同时也包括肾脏病变，骨病变，肺结节和淋巴结肿大。DeepLesion多类别病变数据集可以用来开发自动化放射诊断的CADx系统。

数据集地址：

https://nihcc.app.box.com/v/DeepLesion

## 二.挑战赛/比赛数据

### 1. 放射学中的视觉概念提取挑战赛

从几种不同的成像模式（例如CT和MR）手动注释几种解剖结构（例如肾，肺，膀胱等）的放射学数据。它们还提供了一个云计算实例，任何人都可以使用它来根据基准开发和评估模型。

链接：[http](http://www.visceral.eu/):[//www.visceral.eu/](http://www.visceral.eu/)

### 2. 生物医学图像分析的重大挑战赛

一系列生物医学成像挑战赛，通过标准化评估标准，促进新解决方案与现有解决方案之间的更好比较。您也可以创建自己的挑战赛。在撰写本文时，有92个挑战赛提供可下载的数据集。

链接：[http](http://www.grand-challenge.org/)：[//www.grand-challenge.org/](http://www.grand-challenge.org/)

### 3. DREAM挑战赛

DREAM挑战赛提出了关于系统生物学和转化医学的基本问题。由来自各种组织的研究人员社区设计和运营，我们的挑战赛邀请参与者提出解决方案 - 在此过程中促进协作和建立社区。Sage Bionetworks提供专业知识和机构支持，以及通过其Synapse平台应对挑战赛的基础设施。我们共同拥有一个愿景，允许个人和团体公开合作，使&quot;人群的智慧&quot;对科学和人类健康产生最大的影响。

- 数字乳房X线摄影术梦想挑战赛。
- ICGC-TCGA DREAM体细胞突变调用RNA挑战赛（SMC-RNA）
- DREAM Idea挑战赛
- 这些是添加时的积极挑战赛，还有更多过去的挑战赛和即将到来的挑战赛

链接：[http](http://dreamchallenges.org/):[//dreamchallenges.org/](http://dreamchallenges.org/)

出版物：[http://dreamchallenges.org/publications/](http://dreamchallenges.org/publications/)

Here we describe the use of crowdsourcing to specifically evaluate and benchmark features derived from accelerometer and gyroscope data in two different datasets to predict the presence of Parkinson&#39;s Disease (PD) and severity of three PD symptoms: tremor, dyskinesia and bradykinesia.

### 4. Kaggle糖尿病视网膜病变

高分辨率视网膜图像，由临床医生按0-4严重等级注释，用于检测糖尿病视网膜病变。该数据集是完成的Kaggle竞赛的一部分，该竞赛通常是公开数据集的重要来源。

链接：[https](https://www.kaggle.com/c/diabetic-retinopathy-detection)：[//www.kaggle.com/c/diabetic-retinopathy-detection](https://www.kaggle.com/c/diabetic-retinopathy-detection)

### 5. 宫颈癌筛查

在这场讨价还价的比赛中，您将开发算法，根据宫颈图像正确分类子宫颈类型。我们数据集中的这些不同类型的子宫颈都被认为是正常的（非癌症），但由于转化区并不总是可见的，因此一些患者需要进一步检测，而有些患者则不需要。

链接：[https](https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/data)：[//www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/data](https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/data)

### 6. 多发性硬化病变分割

挑战赛2008年。一组脑MRI扫描检测MS病变。

链接：[http](http://www.ia.unc.edu/MSseg/)：[//www.ia.unc.edu/MSseg/](http://www.ia.unc.edu/MSseg/)

### 7. 多模式脑肿瘤分割挑战赛

脑肿瘤磁共振扫描的大数据集。自2012年以来，他们每年都在扩展这一数据集并进行挑战赛。

链接：[http](http://braintumorsegmentation.org/)：[//braintumorsegmentation.org/](http://braintumorsegmentation.org/)

### 8 .Coding4Cancer

美国国立卫生研究院和Sage Bionetworks基金会的一项新举措，旨在应对一系列改善癌症筛查的挑战赛。第一个是数字乳房X线摄影读数。第二个是肺癌检测。尚未启动挑战赛。

链接：[http](http://coding4cancer.org/)：[//coding4cancer.org/](http://coding4cancer.org/)

### 9. 脑电图挑战赛数据集(Kaggle)

- 墨尔本大学AES / MathWorks / NIH癫痫发作预测 - 预测长期人类颅内脑电图记录中的癫痫发作

链接：[https](https://www.kaggle.com/c/melbourne-university-seizure-prediction)：[//www.kaggle.com/c/melbourne-university-seizure-prediction](https://www.kaggle.com/c/melbourne-university-seizure-prediction)

- 美国癫痫学会癫痫发作预测挑战赛 - 预测颅内脑电图记录中的癫痫发作

链接：[https](https://www.kaggle.com/c/seizure-prediction)：[//www.kaggle.com/c/seizure-prediction](https://www.kaggle.com/c/seizure-prediction)

- UPenn和梅奥诊所的癫痫发作检测挑战赛 - 检测颅内脑电图记录中的癫痫发作

链接：[https](https://www.kaggle.com/c/seizure-detection)：[//www.kaggle.com/c/seizure-detection](https://www.kaggle.com/c/seizure-detection)

- 掌握和提升脑电图检测 - 识别脑电图记录的手部动作

链接：[https](https://www.kaggle.com/c/grasp-and-lift-eeg-detection)：[//www.kaggle.com/c/grasp-and-lift-eeg-detection](https://www.kaggle.com/c/grasp-and-lift-eeg-detection)

### 10. MICCAI会议挑战赛

医学图像计算与计算机辅助干预。大多数挑战赛都会被盛大挑战赛等网站所覆盖。您仍然可以在会议网站的&quot;卫星活动&quot;标签下看到所有这些挑战赛。

链接：[http](http://www.miccai.org/ConferenceHistory)：[//www.miccai.org/ConferenceHistory](http://www.miccai.org/ConferenceHistory)

### 11. 国际生物医学成像研讨会（ISBI）

IEEE国际生物医学成像研讨会（ISBI）是一个科学会议，致力于生物医学成像的数学，算法和计算方面，涵盖所有观察范围。大多数挑战赛将列入重大挑战赛。您仍然可以访问每年网站&quot;计划&quot;下的&quot;挑战赛&quot;标签来访问它。

链接：[http](http://biomedicalimaging.org/)：[//biomedicalimaging.org](http://biomedicalimaging.org/)

### 12 连续注册挑战赛（CRC）

连续注册挑战赛（CRC）是受现代软件开发实践启发的肺和脑图像注册的挑战赛。参与者使用开源SuperElastix C ++ API实现他们的算法。挑战赛集中于肺和脑的成对登记，这是临床环境中经常遇到的两个问题。他们收集了7个开放访问数据集和一个私有数据集（3 + 1肺数据集，4个脑数据集）。挑战赛结果将在即将举行的生物医学图像注册研讨会（WBIR 2018）上进行介绍和讨论。

链接：[https](https://continuousregistration.grand-challenge.org/home/)：[//continuousregistration.grand-challenge.org/home/](https://continuousregistration.grand-challenge.org/home/)

### 13. 使用MURA进行骨骼X射线深度学习比赛

MURA（肌肉骨骼X线片）是骨骼X射线的大型数据集。斯坦福大学集团和AIMI中心正在举办一项竞赛，其中算法的任务是确定X射线研究是正常还是异常。该算法在207项肌肉骨骼研究的测试集上进行评估，其中每项研究由6名经过委员会认证的放射科医师单独回顾性标记为正常或异常。其中三位放射科医师用于创建金标准，定义为放射科医师标签的多数投票，另外三位用于获得最佳放射科医师表现，定义为三位放射科医师的最高得分金标准作为事实。挑战赛排行榜是公开托管的，每两周更新一次。

发布于2018年2月，吴恩达团队开源了 MURA 数据库，MURA 是目前最大的 X 光片数据库之一。该数据库中包含了源自14982项病例的40895张肌肉骨骼X光片。1万多项病例里有9067例正常的上级肌肉骨骼和5915例上肢异常肌肉骨骼的X光片，部位包括肩部、肱骨、手肘、前臂、手腕、手掌和手指。每个病例包含一个或多个图像，均由放射科医师手动标记。全球有超过17亿人都有肌肉骨骼性的疾病，因此训练这个数据集，并基于深度学习检测骨骼疾病，进行自动异常定位，通过组织器官的X光片来确定机体的健康状况，进而对患者的病情进行诊断，可以帮助缓解放射科医生的疲劳。

链接：[https://stanfordmlgroup.github.io/competitions/mura/](https://stanfordmlgroup.github.io/competitions/mura/)

**(1)MURA: Large Dataset for Abnormality Detection in Musculoskeletal Radiographs.**


## 三.来自电子健康记录（EHR）的数据

**1. Building the graph of medicine from millions of clinical narratives**
从数百万临床叙述中构建医学图表
从1400万临床记录和260,000名患者中提取医学术语的共现统计数据。
论文：[http](http://www.nature.com/articles/sdata201432):[//www.nature.com/articles/sdata201432
](http://www.nature.com/articles/sdata201432)数据：[http](/C:%5CUsers%5CQHY%5CAppData%5CRoaming%5CMicrosoft%5CWord%5Chttp):[//datadryad.org/resource/doi](http://datadryad.org/resource/doi:10.5061/dryad.jp917) : [10.5061/dryad.jp917](http://datadryad.org/resource/doi:10.5061/dryad.jp917)

(1) **Building the graph of medicine from millions of clinical narratives**.

Finlayson, Samuel &amp; LePendu, Paea &amp; Shah, Nigam. (Scientific data.2014).

Electronic health records (EHR) represent a rich and relatively untapped resource for characterizing the true nature of clinical practice and for quantifying the degree of inter-relatedness of medical entities such as drugs, diseases, procedures and devices. We provide a unique set of co-occurrence matrices, quantifying the pairwise mentions of 3 million terms mapped onto 1 million clinical concepts, calculated from the raw text of 20 million clinical notes spanning 19 years of data.

根据临床笔记计算得出了一组独特的共现矩阵，量化了映射到100万个临床概念上的300万个术语的成对提及，即量化了医学概念之间的关联性。

### 2. Learning Low-Dimensional Representations of Medical Concept

学习医学概念的低维表示：使用索赔数据构建的医学概念的低维嵌入。请注意，本文利用来自数百万临床叙述的医学图表中的数据。
纸张：[http://people.csail.mit.edu/dsontag/papers/ChoiChiuSontag\_AMIA\_CRI16.pdf](http://people.csail.mit.edu/dsontag/papers/ChoiChiuSontag_AMIA_CRI16.pdf)

数据：[https](https://github.com/clinicalml/embeddings):[//github.com/clinicalml/embeddings](https://github.com/clinicalml/embeddings)

(1) **Learning Low-Dimensional Representations of Medical Concepts.**

Choi, Youngduck &amp; Chiu, Chill &amp; Sontag, David. (AMIA Joint Summits on Translational Science proceedings.2016).

We show how to learn low-dimensional representations (embeddings) of a wide range of concepts in medicine, including diseases (e.g., ICD9 codes), medications, procedures, and laboratory tests. We expect that these embeddings will be useful across medical informatics for tasks such as cohort selection and patient summarization. These embeddings are learned using a technique called neural language modeling from the natural language processing community. However, rather than learning the embeddings solely from text, we show how to learn the embeddings from claims data, which is widely available both to providers and to payers. We also show that with a simple algorithmic adjustment, it is possible to learn medical concept embeddings in a privacy preserving manner from co-occurrence counts derived from clinical narratives. Finally, we establish a methodological framework, arising from standard medical ontologies such as UMLS, NDF-RT, and CCS, to further investigate the embeddings and precisely characterize their quantitative properties.

展示了如何学习医学中各种概念的低维度表示, 使用自然语言处理中称为神经语言建模的技术来学习这些嵌入。说明了通过简单的算法调整就有可能以隐私保护的方式从来自临床描述的共现计数中学习医学概念嵌入。

### 3. MIMIC-III

一个可自由访问的重症监护数据库，38,597名患者的匿名重症监护EHR数据库和53,423名ICU入院患者。需要注册。
论文：[http](http://www.nature.com/articles/sdata201635)：[//www.nature.com/articles/sdata201635
](http://www.nature.com/articles/sdata201635)数据：[http](http://physionet.org/physiobank/database/mimic3cdb/)：[//physionet.org/physiobank/database/mimic3cdb/](http://physionet.org/physiobank/database/mimic3cdb/)

**4. Clinical Concept Embeddings Learned from Massive Sources of Medical Data**
从医学大规模数据的人士处获悉临床概念曲面嵌入
曲面嵌入为108477个医学概念60万名患者，170万篇期刊论文，以及20万名患者的临床笔记

论文：[https://arxiv.org/abs/1804.01486
](https://arxiv.org/abs/1804.01486)


## 四.美国国家医疗保健数据

**1.** **疾病控制和预防中心（ CDC ）**
CDC在许多领域的数据，包括：

- Biomonitoring
- Child Vaccinations
- Flu Vaccinations
- Health Statistics
- Injury &amp; Violence
- MMWR
- Motor Vehicle
- NCHS
- NNDSS
- Pregnancy &amp; Vaccination
- STDs
- Smoking &amp; Tobacco Use
- Teen Vaccinations
- Traumatic Brain Injury
- Vaccinations
- Web Metrics

Landing page: https://data.cdc.gov

 Data Catalog: [https://data.cdc.gov/browse](https://data.cdc.gov/browse)

### 2. Medicare Data

来自医疗保健和医疗补助服务中心（CMS）的医疗保险数据数据，用于医院，疗养院，医生，家庭医疗保健，透析和设备提供商。
 Landing page: https://data.medicare.gov
 Explorer: [https://data.medicare.gov/data](https://data.medicare.gov/data)

**(1)Short-Term Mortality Prediction for Elderly Patients Using Medicare Claims Data.**

Makar, Maggie &amp; Ghassemi, Marzyeh &amp; Cutler, David &amp; Obermeyer, Ziad. (International Journal of Machine Learning and Computing. 2015).

Here we tested a number of machine learning classifiers for prediction of six-month mortality in a population of elderly Medicare beneficiaries, using an administrative claims database of the kind available to the majority of health care payers and providers. We show that machine learning classifiers substantially outperform current widely-used methods of risk prediction-but only when used with an improved feature set incorporating insights from clinical medicine, developed for this study. Our work has applications to supporting patient and provider decision making at the end of life, as well as population health-oriented efforts to identify patients at high risk of poor outcomes.

从临床医学的角度构建机器学习模型来预测死亡率。

### 3. Texas Public Use Inpatient Data File

德克萨斯州公共使用住院患者数据文件：包括2006年至2009年德克萨斯州诊断，程序代码和结果的1100万住院患者就诊情况。

链接：[https](https://www.dshs.texas.gov/thcic/hospitals/Inpatientpudf.shtm):[//www.dshs.texas.gov/thcic/hospitals/Inpatientpudf.shtm](https://www.dshs.texas.gov/thcic/hospitals/Inpatientpudf.shtm)

## 五. UCI数据集

### 1. Liver Disorders Data Set 肝脏疾病数据集
 Data: https://archive.ics.uci.edu/ml/datasets/Liver+Disorders

### 2. Thyroid Disease Data Set 甲状腺疾病数据集

Data: https://archive.ics.uci.edu/ml/datasets/Thyroid+Disease

### 3. Breast Cancer Data Set乳腺癌数据集

Data: [https://archive.ics.uci.edu/ml/datasets/Breast+Cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer)

### 4. Heart Disease Data Set心脏病数据集
 Data: https://archive.ics.uci.edu/ml/datasets/Heart+Disease

### 5. Lymphography Data Set淋巴造影数据集
 Data: https://archive.ics.uci.edu/ml/datasets/Lymphography

### 6. Parkinsons Data Set 帕金森数据集
 Data: https://archive.ics.uci.edu/ml/datasets/parkinsons

### 7. Parkinsons Telemonitoring Data Set帕金森远程监控数据集
 Data: https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring

### 8. Parkinson Speech Dataset with Multiple Types of Sound Recordings Data Set帕金森语音数据集与多种类型的录音数据集
Data: https://archive.ics.uci.edu/ml/datasets/Parkinson+Speech+Dataset+with++Multiple+Types+of+Sound+Recordings

###  9. Parkinson&#39;s Disease Classification Data Set帕金森病分类数据集
 Data: https://archive.ics.uci.edu/ml/datasets/Parkinson%27s+Disease+Classification

## 六. 生物医学文献

###  **1. PMC Open Access**
 Pubmed中心的所有全文，开放访问文章的集合。
 Information: http://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/
 Archived files: [http://www.ncbi.nlm.nih.gov/pmc/tools/ftp/#Data\_Mining](http://www.ncbi.nlm.nih.gov/pmc/tools/ftp/#Data_Mining)

### 2. PubMed 200k RCT

Collection of pubmed abstracts from randomized control trials (RCTs). Annotations for each sentence in the abstract are available.

Paper: https://arxiv.org/abs/1710.06071

Data: https://github.com/Franck-Dernoncourt/pubmed-rctPubMed文章的Web API

NLM还提供了用于访问PubMed中生物医学文献的Web API。

获取PubMed文章的说明：[https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/BioC-PubMed/](https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/BioC-PubMed/)（不是全文，只是标题，摘要等）

对于PubMed Central中的文章，获取整篇文章的说明：[https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/BioC-PMC/](https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/BioC-PMC/)

## 七. TREC精准医学/临床决策支持轨道

文本检索会议（TREC）从2014年起开始实施精准医学/临床决策支持。

### 1. 2014临床决策支持跟踪
焦点：检索与回答医疗记录的一般临床问题相关的生物医学文章。
信息和数据：[http](http://www.trec-cds.org/2014.html)：[//www.trec-cds.org/2014.html](http://www.trec-cds.org/2014.html)

### 2. 2015年临床决策支持跟踪
焦点：检索与回答有关医疗记录的一般临床问题相关的生物医学文章。
信息和数据：[http](http://www.trec-cds.org/2015.html)：[//www.trec-cds.org/2015.html](http://www.trec-cds.org/2015.html)

### 3. 2016年临床决策支持跟踪
重点：检索与回答医疗记录的一般临床问题相关的生物医学文章。使用实际电子健康记录（EHR）患者记录代替合成病例。
信息和数据：[http](http://www.trec-cds.org/2016.html)：[//www.trec-cds.org/2016.html](http://www.trec-cds.org/2016.html)

### 4. 2017年临床决策支持跟踪
焦点：向治疗癌症患者的临床医生检索有用的精确医学相关信息。
信息和数据：[http](http://www.trec-cds.org/2017.html)：[//www.trec-cds.org/2017.html](http://www.trec-cds.org/2017.html)

## 八. 医疗语言数据

### **1. TORGO 数据库：来自构音障碍**
的发言者的声学和发音演讲 TORGO 发音障碍的数据库包括对齐的声学和测量的3D发音特征来自扬声器的脑瘫（CP）或肌萎缩侧索硬化症（ALS），这是两个最常见的语言障碍的普遍原因（Kent和Rosen，2004）和匹配的控制。这个名为TORGO的数据库是多伦多大学计算机科学和语言病理学系与多伦多Holland-Bloorview儿童康复医院合作的结果。

信息和数据：[http](http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html)：[//www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html](http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html)

### **2. NKI-CCRT 语料库**：伴随放化疗治疗晚期头颈癌之前和之后的语言清晰度。
 NKI-CCRT语料库与个人听众对55名因头部和颈部癌症治疗的发言者的录音清晰度的判断将被限制科学使用。语料库包含三个评估时刻的语音清晰度的记录和感知评估：治疗前和治疗后（10周和12个月）。通过化学放射疗法（CCRT）进行治疗。

论文：[http](http://lrec.elra.info/proceedings/lrec2012/pdf/230_Paper.pdf)：[//lrec.elra.info/proceedings/lrec2012/pdf/230\_Paper.pdf](http://lrec.elra.info/proceedings/lrec2012/pdf/230_Paper.pdf)

### 3. 非典型影响Interspeech子挑战赛

BjörnSchuller，Simone Hantke及其同事正在提供EMOTASS语料库。这种独特的语料库是第一个提供来自残疾人的情感语音录音的录音，其中包括更广泛的精神，神经和身体残疾。它包括15名残疾成年人的录音（年龄范围为19至58岁，平均年龄为31.6岁）。任务将是面对非典型展示的五种情绪的分类。录音是在日常工作环境中进行的。总体而言，包括大约11k的话语和大约9个小时的演讲。

论文：[http](http://emotion-research.net/sigs/speech-sig/is2018_compare.pdf):[//emotion-research.net/sigs/speech-sig/is2018\_compare.pdf](http://emotion-research.net/sigs/speech-sig/is2018_compare.pdf)

链接：[http](http://emotion-research.net/sigs/speech-sig/is18-compare)：[//emotion-research.net/sigs/speech-sig/is18-compare](http://emotion-research.net/sigs/speech-sig/is18-compare)。

### 4 .自闭症子挑战赛

自闭症子挑战赛基于&quot;儿童病理语音数据库&quot;（CPSD）。它提供了位于法国巴黎的两所大学儿童和青少年精神病学系（大学Pierre et Marie Curie / Pitie Salpetiere医院和Universite Rene Descartes / Necker医院）的录音。Sub-Challenge中使用的数据集包含来自99名6至18岁儿童的2.5 k语音录音实例

论文：[http](http://emotion-research.net/sigs/speech-sig/is2013_compare.pdf)：[//emotion-research.net/sigs/speech-sig/is2013\_compare.pdf](http://emotion-research.net/sigs/speech-sig/is2013_compare.pdf)

链接：[http](http://emotion-research.net/sigs/speech-sig/is13-compare)：[//emotion-research.net/sigs/speech-sig/is13-compare](http://emotion-research.net/sigs/speech-sig/is13-compare)

### 5. CliCR

A Dataset of Clinical Case Reports for Machine Reading Comprehension

本文提出了一个医疗领域的机器理解（Machine Reading Comprehension

）数据集。该数据集基于大量临床病例报告，对病例进行了约100,000次间隙填充查询。

论文链接：https://www.paperweekly.site/papers/1790

数据集链接：https://github.com/clips/clicr

## 资料

https://github.com/beamandrew/medical-data
