package br.com.cvinicius.ia.vendas;

import java.util.ArrayList;
import java.util.Arrays;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class ApplicationProgrammatic{
	
	public static void main(String[] args) 
	throws Exception{
	
		//Conjunto de dados
		Instances instances = createArff();
		
		//Classificador da aprendizagem gasta_muito ==> [0, 3]
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
				
		//Análise da execução
		double result[] = nb.distributionForInstance(novoItem);
		
		System.out.println("Não: "+result[0]+" ==>> "+result[0] * 100+" %");
		System.out.println("Sim: "+result[1]+" ==>> "+result[1] * 100+" %");
	}
	
	/**
	 * Criação do arquivo .arff através de codigo Java
	 * 
	 * @return Instances
	 */
	public static Instances createArff(){
		
		String arr[] = {"F,>=40,Nao,Nao", "F,20-39,Sim,Sim", "F,>=40,Nao,Sim", "M,20-39,Nao,Sim",	
						"F,>=40,Sim,Sim", "M,20-39,Sim,Nao", "F,>=40,Nao,Nao", "F,>=40,Nao,Sim", 
						"M,>=40,Sim,Nao", "M,20-39,Nao,Sim", "F,>=40,Sim,Nao", "F,20-39,Sim,Nao", 
						"F,20-39,Sim,Nao", "F,20-39,Nao,Sim", "F,20-39,Nao,Sim"};
		
		ArrayList<Attribute> attrs = new ArrayList<>();
		
		attrs.add(new Attribute("sexo", Arrays.asList("F", "M")));
		attrs.add(new Attribute("idade", Arrays.asList("20-39", ">=40")));
		attrs.add(new Attribute("filhos", Arrays.asList("Nao", "Sim")));
		attrs.add(new Attribute("gasta_muito", Arrays.asList("Nao", "Sim")));
		
		Instances instances = new Instances("vendas", attrs, 0);
		
		for(int i=0; i < arr.length; i++){
			
			String[] reg = arr[i].split(",");
						
			Instance item = new DenseInstance(instances.numAttributes());
			
			item.setValue(instances.attribute(0), reg[0]);
			item.setValue(instances.attribute(1), reg[1]);
			item.setValue(instances.attribute(2), reg[2]);
			item.setValue(instances.attribute(3), reg[3]);
			
			instances.add(item);
		}
		
		return instances;
	}
}