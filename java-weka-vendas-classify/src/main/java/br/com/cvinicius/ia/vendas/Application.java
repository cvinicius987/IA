package br.com.cvinicius.ia.vendas;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Application{
	
	public static void main(String[] args) 
	throws Exception{
	
		//Conjunto de dados
		DataSource ds = new DataSource("vendas.arff");
		
		Instances instances = ds.getDataSet();
		
		//Classificador da aprendizagem gasta_muito
		instances.setClassIndex(3);
		
		//Algoritmo utilizado na analise
		NaiveBayes nb = new NaiveBayes();
		
		nb.buildClassifier(instances);
		
		//Novo registro
		Instance novoItem = new DenseInstance(4);
		
		novoItem.setDataset(instances);
		novoItem.setValue(0, "M");
		novoItem.setValue(1, "20-39");
		novoItem.setValue(2, "Sim");
		
		//Analise da execução
		double result[] = nb.distributionForInstance(novoItem);
		
		System.out.println("Não: "+result[0]+" ==>> "+result[0] * 100+" %");
		System.out.println("Sim: "+result[1]+" ==>> "+result[1] * 100+" %");
	}
}