openideo_idetc_2014
===================

Code for our 2014 ASME paper - "How Online Design Communities Evolve Over Time: The Birth and Growth of OpenIDEO"

To replicate paper experiments, run the following from any python prompt: python paper_experiments.py

This code depends on some other python packages, including the following: NumPy MatPlotLib SciPy Simplejson Prettyplotlib brewer2mpl cPickle

Feel free to use all or portions of the data and code for your research so long as you cite the two following articles:

Mark Fuge, Alice Agogino. "How Online Design Communities Evolve Over Time: the Birth and Growth of OpenIDEO," in Proceedings of ASME 2014 International Design Engineering Technical Conferences & Computers and Information in Engineering Conference, August 17-20, 2014, Buffalo, USA.

    @inproceedings{fuge:openideo_evolution_IDETC_2014,
        author = {Mark Fuge and Alice Agogino},
        title = {How Online Design Communities Evolve Over Time: the Birth and Growth of {OpenIDEO}},
        booktitle = {ASME International Design Engineering Technical Conferences},
        year = {2014},
        month = {August},
        location = {Buffalo, USA},
        publisher = {ASME}
    }
    
Mark Fuge, Kevn Tee, Alice Agogino, and Nathan Maton
Analysis of Collaborative Design Networks: A Case Study of OpenIDEO
Journal of Computing and Information Science in Engineering, 14(2) :021009+
(2014, March) 
    
    @article{fuge:openideo_JCISE_2014,
        author = {Fuge, Mark and Tee, Kevin and Agogino, Alice and Maton, Nathan},
        doi = {10.1115/1.4026510},
        issn = {1530-9827},
        journal = {Journal of Computing and Information Science in Engineering},
        month = mar,
        number = {2},
        pages = {021009+},
        title = {Analysis of Collaborative Design Networks: A Case Study of {OpenIDEO}},
        url = {http://dx.doi.org/10.1115/1.4026510},
        volume = {14},
        year = {2014}
    }
    
The code is licensed under the Apache v2 license. 

Note: the raw data files themselves are located in a separate repository, since they serve multiple experiments and it doesn't make sense to duplicate them in multiple places:
https://github.com/IDEALLab/openideo_network_data
    
    
Usage
-----

As an example of how to use our data in your own projects, you can access the concept and social graphs using the following example. It assumes that you have placed the data in a default "graph_data" folder (though this can be changed by passing an argument into load_challenge):
  
    from graph2 import *
    challenge = load_challenge(7)
    # The concept graph is a NetworkX object with concepts as nodes
    # and "built upon" citations as directed edges
    challenge.concept_graph
    # The social graph is a NetworkX object with users as nodes
    # and "communication" actions as directed edges
    challenge.social_graph
    # The link attribute points you to the original challenge site
    challenge.link
