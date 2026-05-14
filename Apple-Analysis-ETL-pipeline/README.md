# Apple Analysis ETL Pipeline using PySpark

## Project Overview

This project implements a modular ETL (Extract, Transform, Load) pipeline using PySpark and Delta Lake in Databricks.

The pipeline analyzes customer purchasing behavior using Spark transformations and workflow orchestration.

---

## Technologies Used

- PySpark
- Databricks
- Delta Lake
- Python
- Spark SQL
- GitHub

---

## Project Architecture

Raw Data
↓
Extractor Layer
↓
Reader Factory
↓
Transformation Layer
↓
Loader Layer
↓
Delta Tables
↓
Analytics

---

## Project Structure

Apple-Analysis-ETL-pipeline/

configs/
readers/
extractor/
transformations/
LoadFactory/
Loader/
main/

---

## Business Use Cases

### 1. Customers who bought AirPods after buying iPhone

Implemented using:
- Window Functions
- lead()
- Broadcast Join

### 2. Customers who bought ONLY AirPods and iPhone

Implemented using:
- groupBy
- collect_set
- array_contains
- Aggregations

---

## Spark Concepts Used

- Broad Transformations
- Narrow Transformations
- Broadcast Join
- Window Functions
- Delta Lake
- Factory Pattern
- Workflow Orchestration

---

## Workflow Architecture

MainETL
↓
Extractor
↓
Transformer
↓
Loader
↓
Delta Sink

---

## Author

Shreyas T S
