<div id="header" align="center">
  <h1>
  🏆 MICELL - The World's Largest Image Dataset for Microscopy Human Body Fluid Test
  <img src="https://media.giphy.com/media/hvRJCLFzcasrR4ia7z/giphy.gif" width="30px"/>
</h1>
</div>

# Overview

Over the past few years, I have collected 300k+ e-Microscope images for the purpose of developing an experimental clinic test AI platform that can be applied more easily and cheaply.
These include erythrocyte, leukocyte, platelet and sperm burns from peripheral blood and frontal fluids specimens.
Dozens of clinical test experts have fully labeled this dataset for object detection and classification.
This is the world's largest e-Microscope image dataset for clinical body fluids test.
I would like to know if someone is willing to exclusively purchase my Micell dataset.
If I can't find a buyer, I'm going to open this dataset together with the report to the world.
Prior to this, I'm going to train object counting and classifying models(Yolo-like) on Micell.
Trained models will be published along with the report and dataset.

# Shot Device

BX53 Biological Microscope (Gumiho)

<span>
  <img src="img/microscope_1.png" width="150px" height="150px"/>
  <img src="img/microscope_2.png" width="150px" height="150px"/>
</span>

# Data Structure

## Platelet
- Detection Dataset (640x480)
<div>
<span>
  <img src="img/platelet/000045.jpg" width="100px" height="100px"/>
  <img src="img/platelet/000047.jpg" width="100px" height="100px"/>
  <img src="img/platelet/000049.jpg" width="100px" height="100px"/>  
</span>
<div>

- Detection Dataset (detect_1920x1080)
<div>
<span>
  <img src="img/platelet/000059.jpg" width="100px" height="100px"/>
  <img src="img/platelet/000061.jpg" width="100px" height="100px"/>
  <img src="img/platelet/000063.jpg" width="100px" height="100px"/>
</span>
<div>

## Prostate
- Detection Dataset (1920x1080)
<div>
<span>
  <img src="img/prostate/000079.jpg" width="100px" height="100px"/>
  <img src="img/prostate/000081.jpg" width="100px" height="100px"/>
</span>
<div>

- Classification Dataset (Small Pieces)
<div>
<span>
  <img src="img/prostate/000093.jpg" width="100px" height="100px"/>
  <img src="img/prostate/000095.jpg" width="100px" height="100px"/>
</span>
<div>

## Red Cell
- Detection Dataset (640x480)
<div>
<span>
  <img src="img/redcell/000005.jpg" width="100px" height="100px"/>
  <img src="img/redcell/000017.jpg" width="100px" height="100px"/>
</span>
<div>

- Detection Dataset (detect_1920x1080)
<div>
<span>
  <img src="img/redcell/000019.jpg" width="100px" height="100px"/>
  <img src="img/redcell/000033.jpg" width="100px" height="100px"/>
</span>
<div>

## White Cell
- Countint Dataset
  DataSet1 (1920x1080)
  DataSet2 (1920x1080)
  Binary Classification Dataset (Small Pieces)

<div>
<span>
  <img src="img/whitecell/000033.jpg" width="100px" height="100px"/>
  <img src="img/whitecell/000077.jpg" width="100px" height="100px"/>
  <img src="img/whitecell/000079.jpg" width="100px" height="100px"/>
</span>
<div>

- Percentage Dataset
  DataSet1 (1920x1080)
  DataSet2 (1920x1080)
  Classification Dataset (Small Pieces for 10 Classes)

<div>
<span>
  <img src="img/whitecell/000081.jpg" width="160px" height="90px"/>
  <img src="img/whitecell/000091.jpg" width="160px" height="90px"/>
  <img src="img/whitecell/000093.jpg" width="160px" height="90px"/>
</span>
<div>

# Labeling
- Exact Positions for Detection Dataset
<div>
<span>
  <img src="img/sam-1.jpg" width="640px" height="360px"/>
  <img src="img/sam-2.jpg" width="640px" height="360px"/>
</span>
<div>

- Exact Positions & Class Indice for Classification Dataset
<div>
<span>
  <img src="img/sam-3.jpg" width="640px" height="360px"/>
</span>
<div>

# Training

With Yolo v7, we achieved 98.78% averaged accuracy.
