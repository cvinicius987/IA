package br.com.cvinicius.ia.vendas;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ApplicationPhone{
	
	public static void main(String[] args) 
	throws Exception{
	
		DataSource ds = new DataSource("phone.arff");
		
		//Conjunto de dados
		Instances instances = ds.getDataSet();
		
		//Classificador da aprendizagem gasta_muito ==> [0, 5]
		instances.setClassIndex(5);
		
		//Algoritmo utilizado na analise
		NaiveBayes nb = new NaiveBayes();
		
		nb.buildClassifier(instances);
		
		//Novo registro
		Instance novoItem = new DenseInstance(6);
		
		novoItem.setDataset(instances);
		novoItem.setValue(0, "62991520209");
		novoItem.setValue(1, "COBRANCA");
		novoItem.setValue(2, "BAD");
		novoItem.setValue(3, "WEDNESDAY");
		novoItem.setValue(4, "12-18");
			
		//Análise da execução
		double result[] = nb.distributionForInstance(novoItem);
		
		System.out.println("S: "+result[0]+" ==>> "+result[0] * 100+" %");
		System.out.println("N: "+result[1]+" ==>> "+result[1] * 100+" %");
	}
}