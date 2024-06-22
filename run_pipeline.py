from pipelines import training_pipeline as tp

if __name__ == "__main__":
    #run pipeline
    tp.training_pipelines(data_path='./data/olist_customers_dataset.csv')
