# generative_language_model
Building a model that can leverage transfer learning to create model generated replies to social media posts


## Project Proposal
Section 3, Andre Fernandes & Miguel Jaime

### Objective
Use the BERT architecture to train a generative model that can initially reply to simulated social media posts (Reddit), then add an active learning component to allow this model to be used for other learning tasks (e.g. learn to respond to tweets or Facebook posts, among other non-social media tasks) with much fewer data examples.

### Motivation
Social media posts tend to use colloquial language and are shorter in nature. This presents an interesting opportunity for modeling: model-generated responses not only need to be intelligible, but the language cannot be too formal, otherwise the posts will look inorganic and out of place. 

If we are able to generate organic-looking social media posts, we could explore their potential impact on improving social media discourse: can friendly, conciliatory, or uplifting messages have a significant effect on subsequent posts? Can we use our model to simulate the effects of positive messaging to an online discussion? If successful, what other ways can we use our quicker, active learning model and infrastructure to allow our model to rapidly learn other tasks with much less data than usual.

### Dataset
We will use a dataset from Reddit comments from May 2015. There are larger and/or newer Reddit datasets, but this one is (a) already clean, and (b) not too large, so we can iterate on our models without waiting long periods of time for training. If needed, we can draw new data from the larger data repositories.

https://www.kaggle.com/reddit/reddit-comments-may-2015#reddit-comments-may-2015.7z

### Algorithms
Neural networks, BERT

### Related Work
BERT [Devlin et al., 2019]
GloVe [Pennington et al., 2014]
Debiasing Word Embeddings [Bolukbasi et al., 2016]
Active Learning [Sculley, 2007]
