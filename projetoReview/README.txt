=============================PROJETO RESTAURANTES=============================
-Overview: Temos os reviews dos restaurantes e  temos o rating que as pessoas deram. A tarefa é tentar prever
o rating com base no que a pessoa escreveu. É uma tarefa que tem como base o campo de estudo de Natural Language
Processing, mais especificamente a área de Sentiment Analysis. Descobrindo quais sentimentos há naquele review, o 
algoritmo vai conseguir correlacionar esses com o rating e, com o modelo treinado, vai conseguir prevê-lo.
 
-Tutoriais:

	-https://github.com/adashofdata/nlp-in-python-tutorial
	-https://github.com/FeatureLabs/predict-restaurant-rating
	-https://nlp.stanford.edu/sentiment/index.html   -------------------------------------------------        # stanford sentiment analysis with paper
	-http://www.libras.ufsc.br/colecaoLetrasLibras/eixoFormacaoBasica/estudosLinguisticos/assets/317/TEXTO_BASE_-_VERSAO_REVISADA.pdf    #linguistica
	-https://machinelearningmastery.com/develop-word-based-neural-language-models-python-keras/        ## why use it if we got elmo?? ##
	https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/     ##                             ##

-Etapas:

	-Gerar dataset via web scraping: ok
	-Análise descritiva: ok
	-Teoria básica:
	-Testar modelos: 
	-Relatórios, slides, etc: 

-Libraries: 

	-http://www.nltk.org/howto/portuguese_en.html
	-textblob #lexicon based model

-Ideias:

	-Bert : https://github.com/google-research/bert
		http://jalammar.github.io/illustrated-bert/ 
		https://github.com/neuralmind-ai/portuguese-bert & 
		https://jalammar.github.io/illustrated-transformer/
		https://arxiv.org/pdf/1511.01432.pdf
		MODELOS PRONTOS: https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb
			         https://towardsdatascience.com/bert-sentiment-analysis-of-app-store-review-db68f7721d94
				 https://github.com/duyunshu/bert-sentiment-analysis
				 https://iust-deep-learning.github.io/972/static_files/project_reports/sent.pdf #modelo em persa

	-Elmo: https://allennlp.org/elmo
	       https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/

	-Electra: https://ai.googleblog.com/2020/03/more-efficient-nlp-model-pre-training.html

	-LER, EXPLICA DO BERT E DO WORD2VEC: https://pathmind.com/wiki/word2vec

	-NeuroNER: http://neuroner.com/
	-Historia dos modelos: https://ruder.io/nlp-imagenet/


-Feitos com o Bert:

	-https://github.com/duyunshu/bert-sentiment-analysis
	-https://medium.com/southpigalle/how-to-perform-better-sentiment-analysis-with-bert-ba127081eda
	-https://www.blog.google/products/search/search-language-understanding-bert/

-Referências: 

	-Hands On- Machine Learning with python
	-https://towardsdatascience.com/sentiment-analysis-with-python-part-1-5ce197074184
	-Liu, B. (2010).Sentiment analysis and subjectivity.
	-Russell,S. and Norvig,P. Artificial Intelligence - a modern approach, Prentice-Hall, 2013
	-http://www.cs.virginia.edu/~hw5x/Course/TextMining-2019Spring/_site/lectures/
	-MTO IMPORTANTE: https://machinelearningmastery.com/statistical-language-modeling-and-neural-language-models/
	