'''
    Experiment code for exploring the social dynamics of OpenIDEO
    Loads network information sequentially and outputs changing graphs over time
    Mark Fuge 2014 with some code written by Kevin Tee
    
    This experiment code is what is used to the produce the results in
    Mark Fuge, Alice Agogino, "How Online Design Communities Evolve Over Time: 
    the Birth and Growth of OpenIDEO," in Proceedings of ASME 2014 
    International Design Engineering Technical Conferences & Computers and 
    Information in Engineering Conference, August 17-20, 2014, Buffalo, USA.
'''

import os
import sys
import json
import simplejson
import csv
from collections import Counter
import numpy as np
import scipy
import scipy.spatial
import datetime
import cPickle as pickle
from graph2 import *
from classes import *
import efficiency

import matplotlib.pylab as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.colors
import matplotlib.dates
from matplotlib.colors import ColorConverter
from matplotlib.ticker import MaxNLocator,AutoLocator
import prettyplotlib as ppl
import matplotlib.dates as mdates
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import brewer2mpl
from matplotlib.ticker import MaxNLocator,AutoLocator
paired = brewer2mpl.get_map('Paired', 'qualitative', 10).mpl_colors
dark2 = brewer2mpl.get_map('Dark2', 'qualitative', 3).mpl_colors
set3 = brewer2mpl.get_map('Set3', 'qualitative', 3).mpl_colors

almost_black = '#262626'
qualitative = brewer2mpl.get_map('Set2', 'qualitative', 8).mpl_colors
qualitative3 = brewer2mpl.get_map('Set3', 'qualitative', 12).mpl_colors
q3hex = brewer2mpl.get_map('Set3', 'qualitative', 12).hex_colors
paired = brewer2mpl.get_map('Paired', 'qualitative', 12).mpl_colors
plt.rc('axes',color_cycle = paired)
plt.rcParams['patch.edgecolor'] = '#555555'

years    = YearLocator()   # every year
months   = MonthLocator(bymonth=[4,7,10])  # every month
yearsFmt = DateFormatter('%Y')
monthFmt = DateFormatter('%b')

def setfont(fontsize=30):
    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'size' : fontsize}
    plt.rc('font', **font)

def fix_axes(ax=None):
    '''
        Removes top and left boxes
        Lightens text
    '''
    if not ax:
        ax = plt.gca()
    # Remove top and right axes lines ("spines")
    spines_to_remove = ['top', 'right']
    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=9, steps=[1, 2, 5, 10],prune='lower'))
    # Change the labels to the off-black
    ax.xaxis.label.set_color(almost_black)
    ax.yaxis.label.set_color(almost_black)
    
def fix_legend(ax=None):
    if not ax:
        ax = plt.gca()
    light_grey = np.array([float(248)/float(255)]*3)
    legend = ax.get_legend()
    ltext = legend.get_texts()
    for lt in ltext:
        plt.setp(lt, color = almost_black)
    rect = legend.get_frame()
    rect.set_facecolor(light_grey)
    rect.set_linewidth(0.0)
    
def check_user(user,user_cache,social_graph,times,event_date):
    if(user not in user_cache):
        try:
            # If we don't have when they joined this will just be whenever we
            # first see commenting activity from the user
            if user not in social_graph.node or 'joined' not in social_graph.node[user]:
                return False
            user_cache[user]=social_graph.node[user]
            joined = user_cache[user]['joined'] # else event_date
            times.append([joined, 'j',(user)])
            user_cache[user]['joined']=joined
            user_cache[user]['lastaction']=event_date
        except Exception:
            raise
    if event_date>user_cache[user]['lastaction']:
        user_cache[user]['lastaction']=event_date
    if (user_cache[user]['joined'] >= event_date):
        # In the unlikely event where this happens, we need to do this to ensure
        # that temporal order of events is correct
        user_cache[user]['joined'] = event_date-datetime.timedelta(0,1) # Minus 1 second
        uind=int(np.where(np.array(times)[:,2]==user)[0])
        # reset the join time:
        times[uind][0]=user_cache[user]['joined'] 
    return True
 

#Whole Network
def centralization(graph,centrality_measure=nx.degree):
    scores = centrality_measure(graph).values()
    max_score = max(scores)
    star = centrality_measure(nx.star_graph(len(graph.nodes())-1)).values()
    max_star = max(star)
    sum_of_differences=0
    for score in scores:
        sum_of_differences+= max_score-score
    max_possible = 0
    for score in star:
        max_possible += max_star-score
    return sum_of_differences/float(max_possible)

 
unwanted_concept_keys=[u'image', u'comments', u'comment_count']

def build_times(challenge,times,user_cache,concept_cache):
    ''' Updates a numpy array 'times' with a timestamp and a tuple of
        information corresponding to the type of event:
         - [time,'j',(user)] = user joined
         - [time,'s',(user,concept)] = user submited a particular concept
         - [time,'c',(userfrom,userto,concept) =user comments to another user
                optionally on a given concept
    '''
    concept_graph = challenge.concept_graph
    social_graph = challenge.social_graph
    challenge_name = challenge.title
    print "Building Social Graph..."
    # Go through all of the concepts
    for concept in concept_graph.node.values():
        this_user = concept['creator']
        concept_cache[concept['id']]=dict(concept)
        concept_cache[concept['id']]['challenge']=challenge_name
        if ('evaluations' in concept_cache[concept['id']] and
            concept_cache[concept['id']]['evaluations']):
            # Could calculate the average here, but for now I'm just going to 
            # use a flag
            concept_cache[concept['id']]['evaluations']=True
        for key in unwanted_concept_keys:
            concept_cache[concept['id']].pop(key,None)
        if not check_user(this_user,user_cache,social_graph,times,concept['created']):
            continue
        times.append([concept['created'],
                      's',
                      (this_user,concept['id'])
                      ])
        # Iterate through that concepts's comments
        if 'comments' not in concept:
            continue
        for comment in concept['comments']:
            # Find the user who made the comment
            commenting_user=comment.user
            comment_date=datetime.datetime.strptime(comment.date,'%B %d, %Y, %I:%M%p')
            # In the case where the comment time and creation time are the same,
            # shift the comment so that it appears
            # after the concept creation when sorted in time
            if comment_date == concept['created']:
                comment_date+=datetime.timedelta(0,1) # Adds 1 second
            if not check_user(commenting_user,user_cache,social_graph,times,comment_date):
                continue
            if comment_date<concept['created']:
                # This is likely some data error
                raise
            times.append([comment_date,
                          'c',
                          (commenting_user,this_user,concept['id'])])
            # Was there a reply? If so, add more weight between the commenting
            # person and the replying person
            if type(comment.sub_comments)!=list:
                continue
            for subcomment in comment.sub_comments:
                replying_user=subcomment.user
                subcomment_date=datetime.datetime.strptime(subcomment.date,'%B %d, %Y, %I:%M%p')
                if not check_user(replying_user,user_cache,social_graph,times,subcomment_date):
                    continue
                if subcomment_date<concept['created']:
                    continue
                times.append([subcomment_date,
                              'c',
                              (replying_user,commenting_user,concept['id'])
                              ])

def update_weights(G,t,lamb=0.046,thresh=1E-4):
    ''' Updates the edge weights by using a power decay formula:
        \sum_i w_i exp(-\lambda|t-t_i|/w_i)
        since w_i = 1, this means that it is really:
        \sum_i exp(-\lambda|t-t_i|)
        bigger lambda = faster decay
        lambda = 0.046 -> link strength is 1% of original after 100 days
        with a cutoff threshold of 0.01 this would mean that links disappear
        after 100 days of inactivity.
    '''
    edges_to_remove=[]
    for u1,u2 in G.edges_iter():
        edge=G[u1][u2]
        new_weight=0
        weights_to_remove=[]
        for weight,wt in edge['weights']:
            deltat=abs(t-wt)
            add_weight = weight*np.exp(-(lamb*deltat.days)/weight)
            new_weight+= add_weight
            # if weight is less that threshold, we should remove it
            if add_weight < thresh:
                # Save on computation by removing very low weight edges
                weights_to_remove.append([weight,wt])
        for w in weights_to_remove:
            edge['weights'].remove(w)
        edge['weight']=new_weight
 
def find_events(challenges,times):
    ''' Locates major temporal events in challenges.
        For example:
         - join dates of important individuals
         - when challanges start and stop
         - 
    '''
    inspiration_start=[]
    concepting_start=[]
    inspiration_end=[]
    concepting_end=[]
    people_join=[]
    for challenge in challenges:
        c=challenge.concept_graph
        types=nx.get_node_attributes(c,'type')
        type_list=np.array(types.items())
        for ntype,scont,econt in [('inspiration',inspiration_start,inspiration_end),
                                  ('concept',concepting_start,concepting_end)]:
            i_ind=type_list[:,1]==ntype
            first_node=None
            first_node_created=datetime.datetime.today()
            last_node_created=datetime.datetime(1,1,1)
            for node in np.array(types.items())[i_ind,0]:
                if c.node[node]['created']<first_node_created:
                    first_node=node
                    first_node_created=c.node[node]['created']
                if c.node[node]['created']>last_node_created:
                    last_node=node
                    last_node_created=c.node[node]['created']
            scont.append([first_node_created,ntype[0]+'c','C%d %s begin'%(challenge.number,ntype[0:3])])
            econt.append([last_node_created,ntype[0]+'e','C%d %s end'%(challenge.number,ntype[0:3])])
    return np.vstack([inspiration_start,#concepting_start,
                      concepting_end])#,people_join])
 
def find_convex_hull(points):
    triangulation = scipy.spatial.Delaunay(points)
    unordered = list(triangulation.convex_hull)
    ordered = list(unordered.pop(0))
    while len(unordered) > 0:
        next = (i for i, seg in enumerate(unordered) if ordered[-1] in seg).next()
        ordered += [point for point in unordered.pop(next) if point != ordered[-1]]
    return points[ordered]
                              
def draw_k_cliques(G,k,cutoff,communities,graph_type,save_path=None,pos=None,
                   challenge_number='',time_string=''):
    #plt.rc('axes',color_cycle = qualitative3)
    challenge_number=str(challenge_number)
    if not pos:
        pos=nx.spring_layout(G.to_undirected())
    node_order=G.node.keys()
    cc = ColorConverter()
    clear = (1, 1, 0, 0.5)
    clear = cc.to_rgba('b',0.1)
    clear='#dddddd'
    node_col =None
    node_col = dict.fromkeys(G.nodes(),clear)
    fig, ax = plt.subplots(frameon=False)
    ax.axis('off')
    for i,com in enumerate(communities):
        com_pnts=[]
        patch_col=q3hex[i%len(q3hex)]
        for person in com:
            com_pnts.append(pos[person])
            if node_col[person]!=clear:
                node_col[person]='r'
            else:
                node_col[person]=patch_col
        if len(np.array(com_pnts))>2:
            try:
                hull_pnts=find_convex_hull(np.array(com_pnts))
                Path = mpath.Path
                path_data = []
                for i,hull_pnt in enumerate(hull_pnts):
                    if i==0:
                        path_data.append((Path.MOVETO, (hull_pnt[0], hull_pnt[1])))
                    else:
                        path_data.append((Path.LINETO, (hull_pnt[0], hull_pnt[1])))
                path_data.append((Path.CLOSEPOLY, (hull_pnt[0], hull_pnt[1])))
                codes, verts = zip(*path_data)
                path = mpath.Path(verts, codes)
                patch = mpatches.PathPatch(path, facecolor=patch_col, alpha=0.5)
                ax.add_patch(patch)
            except Exception as e:
                print " ...Convex Hull Error..."
                print e
    nx.draw_networkx_edges(G,pos=pos,width=1,alpha=0.10)
    nx.draw_networkx_nodes(G,pos=pos,with_labels=False,
                           node_color=[node_col[n] for n in node_order],
                           node_size=50,#node_size=20,
                           alpha=0.9,linewidths=0.1)
    plt.title('%s%s: %d-clique communities'%(graph_type,challenge_number,k))
    plt.tick_params(\
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        left='off',      # ticks along the bottom edge are off
        right='off',         # ticks along the top edge are off
        labelbottom='off', # labels along the bottom edge are off
        labelleft='off')
    if save_path:
        plt.savefig(save_path+'coms_%s_%s_%s_%d-%.4f-clique_communities.pdf'%(graph_type,time_string,challenge_number,k,cutoff))
    else:
        plt.tight_layout()
        fig.show()

def plot_events(ax,major_events):
    event_colors={'ic':'#8bd343',
                  'cc':'#fa9b27',
                  'ie':'#c67a1e',
                  'ce':'#ff6868',
                  'j':'#1cb2d4'}
    bbox_props = dict(boxstyle="round,pad=0.2", fc="w", ec="0.5", alpha=0.0, 
                      lw=0)
    ymin,ymax=ax.get_ylim()
    yrange=ymax-ymin
    for etime,ecode,event_name in major_events:
        etime=matplotlib.dates.date2num(etime)
        ax.axvline(x=etime,label=event_name,color=event_colors[ecode],
                   alpha=0.2,linewidth=2,zorder=-2)
        ax.text(etime, ymax, event_name, ha="center", va="top",
                rotation=90, size=6,color="0.5",
                bbox=bbox_props,zorder=-1)

def plot_user_profile_path(users,user_profile,major_events=None,
                           normalize=False,title="user_activity"):
    ''' For a given list of users, plot their paths and events
    '''
    fig,ax = plt.subplots(tight_layout=True)
    ys=len(users)
    cm=brewer2mpl.get_map('Set2', 'qualitative', 4).get_mpl_colormap()
    cm=brewer2mpl.get_map('Spectral', 'diverging', 4).get_mpl_colormap()
    colordict = {'join':1,'submitted':0,'gotcomment':2,'gavecomment':3}
    profile_json={'user_activity':[]}
    for i,user in enumerate(users):
        events = user_profile[user]
        # Discards getting comments
        events=[e for e in events if e[1]!='gotcomment']
        num_events = len(events)
        timestamps = [e[0] for e in events]
        join_date=timestamps[0]
        if normalize:
            timestamps = [(t-join_date).days for t in timestamps]
        tags = [e[1] for e in events]
        colors = [colordict[t] for t in tags]
        profile_json['user_activity'].append( {'id':user,
                                       # tag  time   # days
                              'events':[{'joinorder':i, 'tag':e[1],
                                         'id': user,
                                         'date':e[0],
                                         'day':(e[0]-join_date).days}
                                        for e in events]
                             })
        ppl.scatter(timestamps,[i]*num_events,s=10, marker='.',linewidth=0,
                    c=colors,cmap=cm,edgecolor='none')
    p1 = plt.Rectangle((0, 0), 1, 1, fc=cm(0/3.))
    p2 = plt.Rectangle((0, 0), 1, 1, fc=cm(1/3.))
    #p3 = plt.Rectangle((0, 0), 1, 1, fc=cm(2/3.))
    p4 = plt.Rectangle((0, 0), 1, 1, fc=cm(3/3.))
    #plt.legend((p1, p2, p3,p4), ('Submitted','Join','GotComment','GaveComment'),
    plt.legend((p1, p2, p4), ('Submitted','Join','GaveComment'),
                loc=1 if normalize else 6,fontsize=10)
    plt.title(title,fontsize=25)
    plt.ylabel("User Rank",fontsize=20)
    xlab = "Days since joining" if normalize else "Time"
    plt.xlabel(xlab,fontsize=20)
    if normalize:
        plt.xlim(xmin=0,xmax=1200)
    else:
        fig.autofmt_xdate()
    # The user count is hard coded here to make comparison of multi-user plots easier
    # This could be algorithmically set in the future
    plt.ylim((0,4000))
    if major_events is not None:
        plot_events(ax,major_events)
        profile_json['major_events']= major_events.tolist()
    fix_legend(ax)
    fig.savefig(figure_path+"profile_%s%s.pdf"%(title,'_norm' if normalize else ''))
    fig.show()
    # Export the JSON Record
    dthandler = lambda obj: obj.isoformat() if isinstance(obj, datetime.datetime) or isinstance(obj, datetime.date) else None
    json.dump(profile_json, open(results_path+'event_%s_data.json'%title,'wb'),
              default=dthandler, sort_keys=True)
      
def plot_transition_matrix(users, user_profile, user_cache, concept_cache, title='',vmax=7.0):
    ''' Plots the autocorrelation between timing events for a given set of users
        Creates a symmetric transitin matrix which can then be plotted, with 
        rows/columns equal to:
        Joining, submitting a concept, giving a comment, getting a comment
    '''
    counts=np.zeros((4,5),dtype=np.int32)
    norm_challenge_counts=np.zeros((4,5),dtype=np.float64)
    ind ={'join':0,'submitted':1,'gotcomment':2,'gavecomment':3,'last':4}
    end_code=ind['last']
    num_users = len(users)
    for user in users:
        num_user_challenges = get_num_user_challenges(user_profile[user],concept_cache)
        last_code=-1
        last_action = user_cache[user]['lastaction']
        for date,code, concept,user2 in user_profile[user]:
            current_code = ind[code]
            if last_code!=-1:
                counts[last_code,current_code]+=1
                norm_challenge_counts[last_code,current_code]+=1.0/num_user_challenges
                if date == last_action:
                    counts[current_code,end_code]+=1
                    norm_challenge_counts[current_code,end_code]+=1.0/num_user_challenges
            last_code=current_code
    counts = counts[:,1:]
    norm_challenge_counts = norm_challenge_counts[:,1:]
    # Normalizes by # users
    user_norm_counts = counts/float(num_users)
    norm_user_challenge_counts=norm_challenge_counts/float(num_users)
    # Save the counts
    count_data = {"raw": counts.tolist(),
                  "user_norm": user_norm_counts.tolist(),
                  "challenge_norm": norm_challenge_counts.tolist(),
                  "user_challenge_norm": norm_user_challenge_counts.tolist()}
    json.dump(count_data, open(results_path+'transition_%s_counts.json'%title,'wb'),
              indent=2, sort_keys=True)
    blues = brewer2mpl.get_map('Blues', 'sequential', 9).get_mpl_colormap()
    labels = ['Join', 'Submit', 'Got Com.', 'Gave Com.','Last']
    fig, ax = plt.subplots(1,tight_layout=True)
    cax = ax.matshow(norm_user_challenge_counts, interpolation='nearest',cmap=blues,
            # This is hard-coded just for the paper figures, so that the color
            # scales mean something
            vmax=vmax)
    fig.colorbar(cax)
    ax.set_yticklabels(['']+labels)
    labels.pop(0)
    ax.set_xticklabels(['']+labels)
    for (i,j), value in np.ndenumerate(norm_user_challenge_counts):
        ax.annotate("%.1f"%(value),xy=(j,i),horizontalalignment='center')
    plt.title(title)
    plt.ylabel('Prior Activity')
    plt.xlabel('Next Activity')
    fig.savefig(figure_path+"transitions_%s.pdf"%(title))
    fig.show()
    
def join_sequence(users, user_profile, filename, num_levels=5, reverse=False):
    ''' Outputs aggregate join sequences for all users '''
    acceptable_events=['join','submitted','gotcomment','gavecomment']
    sequences_hash={}
    for user in users:
        events = user_profile[user]
        events = [e for e in events if e[1] in acceptable_events]
        # Filter out unncessary info
        events = [e[1] for e in events]
        events.append('last')
        if reverse:
            events = events[::-1]
        # Pop the front one
        events.pop(0)
        user_string='-'.join(events[0:num_levels])
        if user_string in sequences_hash:
            sequences_hash[user_string]+=1
        else:
            sequences_hash[user_string]=1
    if reverse:
        filename+="_reverse"
    with open(results_path+'sequence_%s.csv'%filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for key,val in sequences_hash.items():
            writer.writerow([key,val])
            
def get_num_user_challenges(events,concept_cache):
    ''' Returns the number of challenges that this event stream is in '''
    challenge_set=set()
    for date,code, concept,user2 in events:
        if code in ['submitted','gavecomment']:
            current_challenge = concept_cache[concept]['challenge']
            challenge_set.add(current_challenge)
    return len(challenge_set)
    
def get_user_challenges_activity_counts(events,concept_cache):
    ''' Returns the number of challenges that this event stream is in '''
    challenge_set=set()
    activity_dict={}
    for date,code, concept,user2 in events:
        if code in ['submitted','gavecomment']:
            current_challenge = concept_cache[concept]['challenge']
            challenge_set.add(current_challenge)
            pnt=(date, concept, user2)
            if current_challenge not in activity_dict:
                activity_dict[current_challenge]=Counter()
            activity_dict[current_challenge][code]+=1
    return activity_dict

def get_min_activity_level(events, concept_cache):
    ''' Computes the minimum # of activities done across all challenges '''
    activities=get_user_challenges_activity_counts(events,concept_cache)
    return min([sum(a.values()) for a in activities.values()])
    
def get_median_activity_level(events, concept_cache):
    ''' Computes the minimum # of activities done across all challenges '''
    activities=get_user_challenges_activity_counts(events,concept_cache)
    return np.median([sum(a.values()) for a in activities.values()])
    
def get_func_activity_level(events, concept_cache,func):
    ''' Computes a function of activities done across all challenges
        examples of functions: min, np.median, np.mean, scipy.
    '''
    activities=get_user_challenges_activity_counts(events,concept_cache)
    return func([sum(a.values()) for a in activities.values()])

activity_functions = {'high': lambda e: (get_num_user_challenges(e,concept_cache)>=3
                                       and
                                       get_func_activity_level(e, concept_cache,np.median)>=3),
                      'medium': lambda e: (get_num_user_challenges(e,concept_cache)<3
                                       and
                                       get_func_activity_level(e, concept_cache,np.median)>=3),
                      'low': lambda e: (get_num_user_challenges(e,concept_cache)<3
                                       and
                                       get_func_activity_level(e, concept_cache,np.median)<3)
                     }
    
def output_user_activity_summary(users, user_profile, concept_cache):
    ''' Outputs a CSV file with user_ids with some columns relating to activity
        Columns:
            - user_id: the id of the user
            - TODO: user_name: the name of the user
            - num_challenges: the number of challenges they have participated in
            - total_actions: total number of actions they have performed
            - min_action: the minimum number of actions they have performed in 
                          each of the challenges they have participated in
            - median_action: the median number of actions they have performed in 
                             each of the challenges they have participated in
            - mean_action: the mean number of actions they have performed in 
                           each of the challenges they have participated in
            - max_action: the max number of actions they have performed in 
                             any of the challenges they have participated in
            - stddev_action: the standard deviation of the number of actions
                             they have performed in each of the challenges
    '''
    # Set the user functions to evaluate
    funcs = [('Total Actions',sum), ('Min. Actions/challenge',min),
             ('Median Actions/challenge', np.median),
             ('Average Actions/challenge',np.mean),
             ('Max. Actions/challenge',max),
             ('Std. Dev. Actions/challenge',np.std)]
    # 
    with open(results_path+'user_activity_summary.csv','wb') as csv_file:
        fields = ['User ID', '# challenges']
        fields.extend([f[0] for f in funcs])
        csv_writer = csv.writer(csv_file)
        csv_writer = csv.DictWriter(csv_file,fields)
        csv_writer.writeheader()
        for user in users:
            events = user_profile[user]
            info = {'User ID':user,
                    '# challenges': get_num_user_challenges(events,concept_cache)
                    }
            for fname,func in funcs:
                info[fname]=get_func_activity_level(events, concept_cache,func)
            csv_writer.writerow(info)
            
    
def get_users_at_activity_level(users, user_profile, concept_cache, activity_level):
    ''' Goes through activity level and isolates user ids that match a
        particular condition.
        @params
            users: A list of users ids
            user_profile: a dictionary that contains user events
            activity_level: a string representing the desired activity level
    '''
    assert activity_level in activity_functions, "Incorrect key for activity level."
    #- Create function to segment users: for each user, takes in constraints on # submissions/concepts, # challenges
    #-- High Activity: 3+ challenges, with 3+ activities per challenge (both submitting and commenting?)
    #-- Medium Activity: 1-2 challenges, with 3+ activities per challenge
    #-- Low Activity: 1-2 challenges, with <3 activities per challenge
    # Need a function that takes in a user's event history and return boolean
    
    activity_function = activity_functions[activity_level]
        
    # Now interate through the users
    keep_ids=[]
    for user in users:
        if activity_function(user_profile[user]):
            keep_ids.append(user)
    return keep_ids
    
def find_activations(users, user_profile):
    ''' Finds a vector time events that correspond to users becoming activated.
        Here, "activated" means: 
    '''
    # Set the number of days a user has to have no activity before they are
    # considered "inactive"
    num_days_inactive=30
    user_activations={}
    for user in users:
        # Go through the user profile and count the duration between events
        # If that duration is longer than num_days_inactive, its an activation
        last_event=None
        for event in user_profile[user]:
            date,code, concept,user2 = event
            if code in ['gotcomment']:
                continue
            if code != "join" and (date-last_event).days > num_days_inactive:
                # This is an activation
                if user in user_activations:
                    user_activations[user].append(event)
                else:
                    user_activations[user]=[event]
            last_event = date
    return user_activations
            
def get_user_correlates(users, user_profile, major_events):
    ''' Gets possible major events specific to each users that might correlate
        with their participation in the platform. Possible options include:
        - Getting a comment (on your concept vs on another comment)
        - Stage starting (insp., concepting, applause, evaluation, winners)
        - Getting an email? Social Media efforts?
        - A friend/social network neighbor submitting a concept
        - Change in network position?
        - Comments from particular people e.g.,with high network position or DQ
    '''
    # For each particular user, assemble a correlates vector
    
    # This is not the most space efficient, but it straightfoward to analyze
    # I don't anticipate space being an issue here.
    user_correlates={}
    for user in users:
        # received comments are easy, just get this from the user profile
        received_actions =[e for e in user_profile[user] if e[1] in ['gotcomment']]
        user_correlates[user]={'recieved':list(received_actions)}
        # Comments from particular users are doable. Need their user info.
        # I'll do that elsewhere
        user_correlates[user]["major_events"]=major_events.tolist()        
    return user_correlates

def get_activation_matrix(activations,correlates,time_window=5):
    ''' Takes in the user activation levels and builds a Nx2 matrix of counts
        between the correlates and the presence of activations:
                | Activated | Not activated | Total
        Corr. 1 | # events  | # events      | # events
         ...
        Corr. N | # events  | # events      | # events
        Totals  | # events  | # events      | # events
        
        param time_window sets the window of time for an event to count as an 
        activation
    '''
    # Time_window is expressed in days, so convert it to seconds:
    time_window = 3600*time_window
    # Pre-assemble the matrix of counts
    row_keys=correlates[correlates.keys()[0]].keys()
    activation_matrix=np.zeros((len(row_keys),2),dtype=np.int)
    # Now pass through each event and see if it corresponds to a user activation
    for user,correlate_vector in correlates.iteritems():
        for i,row_key in enumerate(row_keys):
            for event in correlate_vector[row_key]:
                date = event[0]
                if user in activations:
                    event_times = np.array([(a[0]-date).total_seconds() for a in activations[user]])
                    sub_activations= np.where((event_times>0)&(event_times<time_window))[0]
                    if(len(sub_activations)>0):
                        # This is an activation
                        act_e=activations[user][sub_activations]
                        print " %s -> %s"%(str((event[1],event[2])),str((act_e[1],act_e[2])))
                        activation_matrix[i,0]+=1
                        continue
                activation_matrix[i,1]+=1            
    return activation_matrix
    
    
def output_activation_vector(activations,correlates, user_profile,
                             pre_time_window=5, post_time_window=30):
    ''' Outputs a JSON vector of activation events to be plotted via dashboard
        pre_time_window: the # of days prior to activation to include
        post_time_window: # of days of activity to include post-activation
        Each row of JSON consists of: user_id, in_events, out_events
        out_events are actions the user takes during the activation window
        in_events are actions that occur in the correlates during the window
    '''
    # time_window is expressed in days, so convert it to seconds:
    #pre_time_window = 3600*pre_time_window
    #post_time_window = 3600*post_time_window
    row_keys=correlates[correlates.keys()[0]].keys()
    # List with one row for each user per activation
    # The same user may occur multiple times if they undergo activation more
    # than once
    user_events = []
    # First get a long list of activation events:
    activation_list = [(u,item) for u,sublist in activations.items() for item in sublist]
    # Now the list is (user,event) for all activations
    # (users may appear in multiple rows)
    
    for user, event in activation_list:
        date,code, concept,user2 = event
        user_info = {"user_id":user}
        # Set the cutoff dates for this activation
        pre_cutoff = date-datetime.timedelta(days=pre_time_window)
        post_cutoff = date+datetime.timedelta(days=post_time_window)
        
        # Do in_events
        user_info['in_events']=[]
        correlate_vector = correlates[user]
        for i,row_key in enumerate(row_keys):
            for event in correlate_vector[row_key]:
                edate = event[0]
                if edate>pre_cutoff and edate < post_cutoff:
                    # include this event
                    # Add the # days to the event signature
                    event = [(edate-date).seconds/3600.]+list(event)
                    user_info['in_events'].append(event)
        # Do out_events
        user_info['out_events']=[]
        for event in user_profile[user]:
            edate,code, concept,user2 = event
            if edate>pre_cutoff and edate < post_cutoff and code not in ['gotcomment']:
                # include this event
                # Add the # days to the event signature
                event = [(edate-date).seconds/3600.]+list(event)
                user_info['out_events'].append(event)
        user_events.append(user_info)
    
    dthandler = lambda obj: obj.isoformat() if isinstance(obj, datetime.datetime) or isinstance(obj, datetime.date) else None
    json.dump(user_events, open(results_path+"activation_info.json",'wb'),
              default=dthandler, sort_keys=True)
    return user_events
    
if __name__ == '__main__':
    setfont(15)
    
    show_events=False
    figure_path = './figures/'
    results_path = './results/'
    challenge_data_path = './graph_data/'
    
    for path in [figure_path, results_path]:
        if not os.path.exists(path):
            os.makedirs(path)
       
    # Load up the data
    max_challenge=22
    print "Loading Challenge Data..."
    challenges = [load_challenge(i,data_path=challenge_data_path) for i in range(max_challenge)]
    
    times=[]
    user_cache={}
    concept_cache={}
    for i,challenge in enumerate(challenges):
        build_times(challenge,
                    times,user_cache,concept_cache)
    times=np.array(times)
    
    # Now sort by time and creation:
    time_index=np.argsort(times[:,0])
    times = times[time_index,:]
    
    # Now define the master graph that will evolve over time--this will include
    # both concepts and users, with different node attributes for each
    # edges will also be given attributes depending on the type of relationship
    G = nx.DiGraph()
    # Just define a sub-social graph that we will update in turn
    socG = nx.DiGraph()
    
    k=6
    cutoff=0.01
    capture_x=[]
    community_x=[]
    num_communities=[]
    user_profile={}
    
    
    largest_component = lambda graph:nx.connected_component_subgraphs(graph)[0]
    # Some of the below functions are disabled, simply to reduce experiment run time
    # Uncomment them if you want those attributes (or add your own to the list!)
    capture_funcs={'Assortativity':nx.degree_assortativity_coefficient,
                   'Clustering':nx.average_clustering,
                   #'Density':nx.density,
                   #'Diameter':lambda x: nx.diameter(largest_component(x)),
                   'Global Efficiency':efficiency.global_efficiency,
                   #'Local Efficiency':efficiency.average_local_efficiency
                   }
    func_data={}
    for name,func in capture_funcs.items():
        func_data[name]=[]
    i=len(times)
    try:
        # First try loading everything from a file first...
        results=pickle.load(open(results_path+'data_k-%d_cutoff-%.3f_i-%d.pickle'%(k,cutoff,i),'rb'))
        major_events = results['major_events']
        community_x = results['community_x']
        capture_x = results['capture_x']
        num_communities = results['num_communities']
        func_data = results['func_data']
        user_profile = results['user_profile']
    except Exception as e:
        # If the saved data file isn't available, then go ahead and do some heavy lifting
        # Running this portion will take some time, but it will save it once, and then you
        # can re-run it again with plotting changes and it will speed up significantly.
        major_events = find_events(challenges,times)
        for i,(time,code,vals) in enumerate(times):
            if code == 'j':
                G.add_node(vals,type='user',created=time)
                socG.add_node(vals,type='user',created=time)
                user_profile[vals]=[(time,'join','','')]
            if code == 's':
                user,concept=vals
                concept_type=concept_cache[concept]['type']
                G.add_node(concept,type=concept_type,created=time)
                G.add_edge(user,concept,type='submitted',
                                      created=time,weight=1,weights=[[1,time]])
                user_profile[user].append((time,'submitted',concept,''))
            if code == 'c':
                fromuser,touser,concept=vals
                G.add_node(fromuser,type='user')
                G.add_node(touser,type='user')
                socG.add_node(fromuser,type='user')
                socG.add_node(touser,type='user')
                try:
                    user_profile[fromuser].append((time,'gavecomment',concept,touser))
                    user_profile[touser].append((time,'gotcomment',concept,fromuser))
                except KeyError as e:
                    # Some occasional temporal issues that need to be re-normalized
                    pass
                    #print e
                    #print "data error with %s->%s for %s"%(fromuser,touser,concept)
                    
                if G.has_edge(fromuser, touser):
                    # we added this one before, just increase the weight by one
                    G[fromuser][touser]['weight'] += 1
                    G[fromuser][touser]['weights'].append([1,time])
                    socG[fromuser][touser]['weight'] += 1
                    socG[fromuser][touser]['weights'].append([1,time])
                else:
                    # new edge. add with weight=1
                    G.add_edge(fromuser, touser,type='comment',
                                          created=time, weight=1,weights=[[1,time]])
                    socG.add_edge(fromuser, touser,type='comment',
                                          created=time, weight=1,weights=[[1,time]])
                if G.has_edge(fromuser, concept):
                    # we added this one before, just increase the weight by one
                    G[fromuser][concept]['weight'] += 1
                    G[fromuser][concept]['weights'].append([1,time])
                else:
                    # new edge. add with weight=1
                    G.add_edge(fromuser, concept,type='comment',
                                          created=time, weight=1,weights=[[1,time]])
            # After a certain number of actions, go ahead and print out some network stats
            if i%100==99:
                
                SG=G.subgraph( [n for n,attrdict in G.node.items() if attrdict['type'] == 'user' ] )
                update_weights(SG,time,lamb=0.046)
                SG_r=nx.DiGraph([(u,v,d) for u,v,d in SG.edges_iter(data=True) if d['weight']>cutoff] )
                
                SG_ru=SG_r.to_undirected()
                # Calculate various functions of interest in "capture_funcs"
                if len(SG_ru)>2:
                    capture_x.append(time)
                    for name,func in capture_funcs.items():
                        func_data[name].append(func(SG_ru))
                
                communities = list(nx.k_clique_communities(SG_ru,k))
                if sum(1 for _ in communities)==0:
                    print "No communities at time %d"%i
                    continue
                else:
                    print "%6d:"%i+'.'*len(communities)
                for community in communities:
                    community_x.append(time)
                    num_communities.append(len(community))     
                # At longer intervals, print out the community pictures,
                # to get a visual sense of things
                if i%1000==999:
                    draw_k_cliques(SG_r,k,cutoff,communities,'Social',
                                   challenge_number=i,
                                   save_path=figure_path,
                                   time_string=time.strftime('%Y-%m-%d'),
                                   pos=nx.spring_layout(SG_ru))
                
        results={'community_x':community_x,
                 'capture_x':capture_x,
                 'num_communities':num_communities,
                 'func_data':func_data,
                 'major_events':major_events,
                 'times':times,
                 'k':k, 'cutoff':cutoff,
                 'user_profile':user_profile
                }
        pickle.dump(results,open(results_path+'data_k-%d_cutoff-%.3f_i-%d.pickle'%(k,cutoff,len(times)),'wb'))
    
    major_events = find_events(challenges,times)
    dates = matplotlib.dates.date2num(np.array(community_x))
    plt.rc('axes',color_cycle = dark2)
    fig,ax=plt.subplots(1,tight_layout=True)
    ppl.plot(dates,num_communities,'.')
    if show_events:
        plot_events(ax,major_events)
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)
    ax.xaxis.set_minor_formatter(monthFmt)
    plt.xticks(rotation=90,fontsize=20)
    plt.setp( ax.xaxis.get_minorticklabels(), rotation=90,fontsize=15)
    plt.title('Community size with k=%d, cutoff=%.3f'%(k,cutoff),fontsize=15)
    plt.ylabel('# Members in Community',fontsize=20)
    fig.savefig(figure_path+"community_size_vs_time_k-%d_cutoff-%.3f.pdf"%(k,cutoff))
    fig.show()
    dates = matplotlib.dates.date2num(np.array(capture_x))
    for name,func in capture_funcs.items():
        fig,ax=plt.subplots(1,tight_layout=True)
        ppl.plot(dates,func_data[name],'.')
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        ax.xaxis.set_minor_locator(months)
        ax.xaxis.set_minor_formatter(monthFmt)
        plt.xticks(rotation=90,fontsize=20)
        plt.setp( ax.xaxis.get_minorticklabels(), rotation=90,fontsize=15)
        if show_events:
            plot_events(ax,major_events)
        plt.title('%s over time'%name,fontsize=25)
        plt.ylabel('%s'%name,fontsize=20)
        fig.savefig(figure_path+"%s_vs_time_k-%d_cutoff-%.3f.pdf"%(name,k,cutoff))
        fig.show()
    # Get some event time histograms
    codes = ['j','s','c']
    names= {'j':'Joins', 's':'Submissions', 'c':'Comments'}
    for code in codes:
        ind = times[:,1]==code
        dates = matplotlib.dates.date2num(times[ind,0])
        fig,ax = plt.subplots(tight_layout=True)
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        ax.xaxis.set_minor_locator(months)
        ax.xaxis.set_minor_formatter(monthFmt)
        ppl.hist(ax,dates,grid='y',bins=int(np.sqrt(len(times[ind,0]))))
        if show_events:
            plot_events(ax,major_events)
        plt.title("%s over time"%names[code])
        plt.ylabel("Frequency",fontsize=20)
        plt.xlim()
        plt.xticks(rotation=90,fontsize=20)
        plt.setp( ax.xaxis.get_minorticklabels(), rotation=90,fontsize=15)
        fig.savefig(figure_path+"%s_vs_time_overall.pdf"%(names[code]))
        fig.show()
        
    # Now gather the user lifetimes
    joins=times[:,1]=='j'
    user_lifetimes=[]
    user_lifetimes_percent=[]
    collect_date=datetime.datetime(2013, 11, 4, 20, 19)
    # Disregard any users that joined during the the last month prior
    # to when I stopped collecting the data, as their lifetime behavior
    # will not be as representative
    check_date = collect_date - datetime.timedelta(days=30)
    for join_date,user in times[joins][:,[0,2]]:
        last_date=user_cache[user]['lastaction']
        lifetime = last_date-join_date
        total_window=check_date-join_date
        user_lifetimes.append((lifetime.total_seconds()/(3600.0*24),user))
        # For viewing
        user_lifetimes_percent.append((100.0*(lifetime.total_seconds()/(3600.0*24))/(total_window.total_seconds()/(3600.0*24)),user))
    dt = np.dtype([('days', np.float32,1),('users', np.str_, 64)])
    user_lifetimes=np.array(user_lifetimes,dtype=dt)
    dt = np.dtype([('lifepercent', np.float32,1),('users', np.str_, 64)])
    user_lifetimes_percent=np.array(user_lifetimes_percent,dtype=dt)
    
    # print out the user lifetimes
    fig,ax = plt.subplots(tight_layout=True)
    ppl.hist(ax,user_lifetimes['days'],grid='y',log=True,
             bins=int(np.sqrt(len(user_lifetimes['days']))))
    plt.title("User Lifetime in Days",fontsize=25)
    plt.ylabel("Frequency",fontsize=20)
    plt.xlabel("Days between Joining and Last Activity",fontsize=20)
    plt.xlim()
    fig.savefig(figure_path+"user_lifetimes.pdf")
    fig.show()
    
    active_ids=user_lifetimes['days']>0
    fig,ax = plt.subplots(tight_layout=True)
    ppl.hist(ax,user_lifetimes_percent['lifepercent'][active_ids],grid='y',
             bins=int(np.sqrt(len(user_lifetimes_percent['lifepercent'][active_ids]))))
    plt.title("User Lifetime in Days",fontsize=25)
    plt.ylabel("Frequency",fontsize=20)
    plt.xlabel("% of window between Joining and Last Activity",fontsize=20)
    plt.xlim(0,100)
    fig.savefig(figure_path+"user_lifetimes_percent.pdf")
    fig.show()
    total_users= len(user_lifetimes['days'])
    total_active= np.count_nonzero(user_lifetimes['days'])     
    # Now need to split users into some groups:
    # Multi-Challenge folks vs. Single Challenge Folks
    # For each user, go through and see if they submitted concepts or gave
    # comments to concepts that were in different challenges
    users=user_lifetimes[active_ids]
    all_users = [u[1] for u in users]
    multi_challenge_rows=[]
    single_challenge_rows=[]
    for i,(days,user) in enumerate(users):
        if get_num_user_challenges(user_profile[user],concept_cache)>1:
            multi_challenge_rows.append(i)
        else:
            single_challenge_rows.append(i)
    single_users=set(users['users'][single_challenge_rows])
    multi_users=set(users['users'][multi_challenge_rows])
    # Joined during a challenge vs. joined not during challenge
    # Get the windows for during a challenge:
    challenge_windows={}
    minds=major_events[:,1]!='j'
    for date, code, inst in major_events[minds]:
        chal_num = int(inst[1:3])   # Only works for up to two digits, which is fine for now
        if chal_num not in challenge_windows:
            challenge_windows[chal_num]=[]
        if code == 'ic' or code == 'ce':
            challenge_windows[chal_num].append(date)
    # Get the joined dates:
    jind=times[:,1]=='j'
    joined_challenge_users=[]
    joined_nochallenge_users=[]
    for join_date, code, user in times[jind]:
        joined_during_challenge = False
        for begin,end in challenge_windows.values():
            if begin>end:
                temp=end; end=begin; begin=temp
            if join_date > begin and join_date<end:
                joined_during_challenge=True
        if joined_during_challenge:
            joined_challenge_users.append(user)
        else:
            joined_nochallenge_users.append(user)
    joined_challenge_users=set(joined_challenge_users)
    joined_nochallenge_users=set(joined_nochallenge_users)

    # Now we can subdivide everyone
    multi_in = list(joined_challenge_users.intersection(multi_users))
    multi_no = list(joined_nochallenge_users.intersection(multi_users))
    single_in = list(joined_challenge_users.intersection(single_users))
    single_no = list(joined_nochallenge_users.intersection(single_users))
    multi_users = list(multi_users)
    single_users = list(single_users)
    # Sort by join date
    multi_in = sorted(multi_in,key=lambda x: user_cache[x]['joined'])
    multi_no = sorted(multi_no,key=lambda x: user_cache[x]['joined'])
    single_in = sorted(single_in,key=lambda x: user_cache[x]['joined'])
    single_no = sorted(single_no,key=lambda x: user_cache[x]['joined'])
    single_users = sorted(single_users,key=lambda x: user_cache[x]['joined'])
    multi_users = sorted(multi_users,key=lambda x: user_cache[x]['joined'])
    #Sort by # contributions
    all_users_contrib_sorted = sorted(all_users,key=lambda x: len(user_profile[x]),reverse=True)
    
    # User Profiles
    # Uncomment lines below for different slices of the dataset
    plot_user_profile_path(all_users, user_profile, normalize=False,title="all_users")
    # plot_user_profile_path(multi_in, user_profile, normalize=True,
                           # title="Multi_in_challenge")
    # plot_user_profile_path(multi_no, user_profile, normalize=True,
                           # title="Multi_no_challenge")
    # plot_user_profile_path(single_in, user_profile, normalize=True,
                           # title="Single_in_challenge")
    # plot_user_profile_path(single_no, user_profile, normalize=True,
                           # title="Single_no_challenge")          
    # plot_user_profile_path(single_users, user_profile, normalize=True,
                           # title="Single_challenge")
    # plot_user_profile_path(multi_users, user_profile, normalize=True,
                           # title="Multi_challenge")        
    # order_all=list(single_users); order_all.extend(multi_users);
    # plot_user_profile_path(order_all, user_profile, normalize=True,
                           # title="order_all_challenge")                            
    # plot_user_profile_path(multi_in, user_profile, normalize=False,
                           # title="Multi_in_challenge",major_events = major_events)
    # plot_user_profile_path(multi_no, user_profile, normalize=False,
                           # title="Multi_no_challenge",major_events = major_events)
    # plot_user_profile_path(single_in, user_profile, normalize=False,
                           # title="Single_in_challenge",major_events = major_events)
    # plot_user_profile_path(single_no, user_profile, normalize=False,
                           # title="Single_no_challenge",major_events = major_events)
    # plot_user_profile_path(single_users, user_profile, normalize=False,
                           # title="Single_challenge",major_events = major_events)
    # plot_user_profile_path(multi_users, user_profile, normalize=False,
                           # title="Multi_challenge",major_events = major_events)
    #plot_user_profile_path(all_users_contrib_sorted, user_profile, normalize=False,title="Sorted")
    plot_user_profile_path(all_users_contrib_sorted, user_profile, normalize=True,title="Sorted")
    # # Transition Matrix
    # plot_transition_matrix(all_users, user_profile, user_cache, concept_cache, title="All_Users")
    # plot_transition_matrix(multi_in, user_profile, user_cache, concept_cache, title="Multi_in_challenge")
    # plot_transition_matrix(multi_no, user_profile, user_cache, concept_cache, title="Multi_no_challenge")
    # plot_transition_matrix(single_in, user_profile, user_cache, concept_cache, title="Single_in_challenge")
    # plot_transition_matrix(single_no, user_profile, user_cache, concept_cache, title="Single_no_challenge")
    plot_transition_matrix(single_users, user_profile, user_cache, concept_cache, title="Single_challenge",vmax=4.0)
    plot_transition_matrix(multi_users, user_profile, user_cache, concept_cache, title="Multi_challenge",vmax=4.0)

    # Remaining open question: What makes the multi challenge folks different
    # than the single challenge folks?
    join_sequence(all_users,user_profile,'all_users',num_levels=5)
    join_sequence(multi_users,user_profile,'multi_users',num_levels=5)
    join_sequence(single_users,user_profile,'single_users',num_levels=5)
    join_sequence(all_users,user_profile,'all_users',num_levels=5,reverse=True)
    join_sequence(multi_users,user_profile,'multi_users',num_levels=5,reverse=True)
    join_sequence(single_users,user_profile,'single_users',num_levels=5,reverse=True)
    
    
    # Now segment users according to activity level:
    output_user_activity_summary(all_users, user_profile, concept_cache)
    high_activity_users=get_users_at_activity_level(all_users, user_profile, 
                                                    concept_cache, 'high')
    medium_activity_users=get_users_at_activity_level(all_users, user_profile, 
                                                    concept_cache, 'medium')
    low_activity_users=get_users_at_activity_level(all_users, user_profile, 
                                                    concept_cache, 'low')
    join_sequence(high_activity_users,user_profile,'high_activity_users',num_levels=5)
    join_sequence(medium_activity_users,user_profile,'medium_activity_users',num_levels=5)
    join_sequence(low_activity_users,user_profile,'low_activity_users',num_levels=5)
    join_sequence(high_activity_users,user_profile,'high_activity_users',num_levels=5,reverse=True)
    join_sequence(medium_activity_users,user_profile,'medium_activity_users',num_levels=5,reverse=True)
    join_sequence(low_activity_users,user_profile,'low_activity_users',num_levels=5,reverse=True)   
    
    plot_transition_matrix(high_activity_users, user_profile, user_cache, concept_cache, title="High_activity")
    plot_transition_matrix(medium_activity_users, user_profile, user_cache, concept_cache, title="Medium_activity")
    plot_transition_matrix(low_activity_users, user_profile, user_cache, concept_cache, title="Low_activity")
    plot_user_profile_path(high_activity_users, user_profile, normalize=True,title="High_activity")
    plot_user_profile_path(medium_activity_users, user_profile, normalize=True,title="Medium_activity")
    plot_user_profile_path(low_activity_users, user_profile, normalize=True,title="Low_activity")
    order_activity=list(high_activity_users);
    order_activity.extend(medium_activity_users);
    order_activity.extend(low_activity_users);
    plot_user_profile_path(order_activity, user_profile, normalize=True,
                           title="order_activity_challenge")
    # Now time to look for event correlations
    # First, need a vector of activation events
    activations = find_activations(all_users, user_profile)
    
    # Then, need a vector of possible correlates
    correlates = get_user_correlates(all_users, user_profile, major_events)
    
    get_activation_matrix(activations,correlates)
    