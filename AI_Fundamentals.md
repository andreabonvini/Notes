# AI Fundamentals

# Search Algorithms

Legenda:

- b=  Branching factor: what is the maximum number of children a node can have, the maximum cardinality of the set returned
- Epsilon is the smallest path cost you have.

| Strategy                   | Complete?                                                    | Optimal?                                                     |                             T C                             | S                                                           |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | :---------------------------------------------------------: | ----------------------------------------------------------- |
| Breadth First Search       | Yes (but not when the branch factor (b) is infinite)         | No<br />Unless<br/>the cost for each node at a level b is greater of all the costs at level b-1
and lower than all the costs at level b+1 (unless the cost function is
increasing with depth) |                           O(b^d)                            | O(b^d)                                                      |
| Uniform Cost               | Yes (but it's not guaranteed when some costs are zero)       | Yes                                                          | O(b^c*/epsilon)<br />con esponente approssimato per eccesso | O(b^c*/epsilon)<br />con esponente approssimato per eccesso |
| Depth First                | not complete if tree search because I can end in a loop<br />complete if I use a graph search (closed list) | No                                                           |     O(b^m)<br />Graph: O(...)<br />Order of the states      | O(bm)                                                       |
| Depth Limited Search       | No                                                           | No                                                           |                          O(\|b^\|)                          | O(bl)                                                       |
| Iterative Deepening Search | Yes ( but not when it's infinite)                            | No (but if all costs are 1 it finds the optimal solution)    |                           O(b^d)                            | O(bd)                                                       |



Comments

- Breadth First Search
  - if we check if a node corresponds to a state that is a goal-state when we generate the node, and not when we pick up the node. This improves the worst case to  O(b^(d+1)).
- Uniform cost Search
  - A property is g(n) that is the path cost. Chooses first from the frontier the node with the smallest path cost. 
    You can never have a loop if costs are different from zero
- Depth First Search
  -  Depth-first strategy: when you have to choose a node from the frontier choose the deepest node. "m" is the longest path i can have in the state space.
  - Backtracking : when you expand a node you don't generate all successors but only one, you keep in a separate data structure a track of which are the successor already tried. 
    A memory saving version of depth first. Here spatial complexity is "m", excluding the data structure. 
- Depth Limited Search
  - I will fix a limit called "l" that is the maximum depth at which i can
    generate successors. I will never generate successors that are deeper than
    "l". i check if the node I'm trying to expand is greater than
    "l", if it is i will not expand.
- Iterative Deepening Search
  - The idea is, size of L is a problem, if it's too little i cannot find any solution.<br />Here we start from l=0, if i find a solution then => good  <br />Every time i keep in memory just one path, just one path at a time.<br />Time complexity: How many nodes I have to generate? (d+1) +bd+b^2x(d-1)+…+b^d

## Admissibility and Consistency

- #### Admissibility of a heuristic:

  The heuristic of a node is admissible iff
  $ h(A) \le MinPathFrom(A) $
  Where the right member is the minimum path to get to the goal state from A

- #### Consistency of a heuristic:

  1 and 2 are two nodes of the graph, h(1) is the heuristic of 1 and h(2) is the heuristic of 2.
  If such nodes are connected in the following way

  ​                                                                                 1 --> 2
  Their heuristics are consistence iff
  $ c_{12}\ge h(1)-h(2) $


## Constraint Satisfaction Problems

##### ARC CONSISTENCY

1. build the constraint directed graph

2. Take a walk in the queue knowing that the arc "X -> Y"  is consistent if for every value of X there is some possible legal value of Y

3. 

   1. 
      if it is not consistent 

      1. I will change the domain of the left member node (X in the example) in order to make it consistent
      2. 
         1. if after removing the element of the domain that makes the arc consistent I obtain an empty domain there is no solution and I stop
         2. otherwise I'll push in the queue some other arcs: 
            since I modified the domain of X, I will have to push in the queue all the arcs in the form of Z -> X, where Z is any node of the graph, excluding Y and excluding the arcs I already evaluated consistent.

   2. 
      if it is consistent 

      1. I throw it away and I forget it forever, it will never be back in the queue
      2. if the queue is empty I stop

4. go back to step 2.



## Inference in propositional logic



#### Definitions

- *model*
  In Propositional Logic, a model is an assignment of truth values to all propositional symbols. 

- *satisfiability*
  a sentence is satisfiable if and only if there is a model that satisfies it. 
  A model satisfies a sentence if the sentence is true under the assignment. 
- *entailment*
  Sentence alpha entails sentence beta if and only if every model of alpha is also a model of beta. 



#### Resolution Inference Procedure

solve
$$
\phi_1 |= \phi_2
$$


1. negate 

$$
\phi_2
$$

2. put both \phi_1 and not\phi_2 in Conjunctive Normal Form (all subformulas divided by a logical AND)
3. Enumerate all clauses
4. compare them together, if a literal appears in both clauses and in only one of them it is negated we get rid of it and write a new clause (A; not A,B -> B)
5. We do not write the new clause if it has already been written
6. the initial expression is true if in the end we obtain the empty clause

##### Strategies for selecting clauses:

- Unit-preference strategy: 
  Give preference to resolutions involving the clauses with the smallest number of literals.
- Set-of-support strategy: 
  Try to resolve with the negated theorem or a clause generated by resolution from that clause.
- Subsumption: 
  Eliminates all sentences that are subsumed (i.e., more specific than) an existing sentence in the KB.

#### DPLL Algorithm

solve 
$$
\phi_1 |= \phi_2
$$

1. Lets draws 3 vertical lines and put in the middle one all the clauses in AND (phi_1 AND not phi_2)

2. giving the precedence to one-literals and then the pure literals (literals that appear just in one form (or all negated or all not negated)), insert them in the knowledge base in order of appearance (left section) specifying in the right section what you've done / observations (If you really need to...).
   Every time that I insert a literal in the knowledge base I exploit such information simplifying the clauses in the mid section (A=1 -> notA or B becomes B)

3. The goal is to end up with the empty clause in the mid section

   P.S.: 

   - if we find ourselves in having no pure literals and no one-literals, we consider one of the remaining literals (for example A) and split the solution in two (s' and s''). s' will consider A in the knowledge base, while s'' will consider notA. 
     phi_1 implies phi_2 only if both parallel solutions return an empty clause (not verified, but seems pretty obvious to me).
   - If we find ourselves with only one logical proposition left, for example 
     (A or B), let's choose just the first one in order of appearance and put it in the knowledge base. this means that B can assume whatever value it wants, it won't influence the result.



# Zero Sum Game

A zero sum game is (confusingly) defined as one where the total payoff to all players is the same for every instance of the game.
Chess is zero sum because every game has payoff of either 0+1, 1+0,or 1/2 +1/2.
"constant-sum" would have been a better term,.



# Alpha-Beta Pruning

- alpha initial value = + infinity 

- beta initial value = - infinity

- v initial value = none.

- alpha & beta are inherited by the daddy

- v is inherited by the children

- the first v is computed on the left leaf (depth first search)
- Pruning condition
  - If the utility function v is bounded --> as soon as we find find a winning path for max we end the search there
  - if the utility function v is not bounded --> if alpha > beta we prune

## Inference

#### Backward Chaining

1. Draw the root of the tree
2. derive the AND children.
3. Analyze the children from the left to the right:
   1. 
      if the AND child is formed of at least a literal that is not a clause in the KB we drop the AND child  and go back to 3.
   2. 
      1. if a literal of the AND child is a clause of the KB we are happy with it and leave the terminal alone.
         if a literal of and AND child is a right member of a clause of the KB go back to 2.
      2. if all the terminals of the AND child are clauses of the KB we are happy with them, leave them alone and go back to 3.

















# A Star algorithm - Optimality:

If the heuristic function *h* is [admissible](https://en.wikipedia.org/wiki/Admissible_heuristic), meaning that it never overestimates the actual minimal cost of reaching the goal, then A* is itself admissible (or *optimal*) if we do not use a closed set. If a closed set is used, then *h* must also be *monotonic* (or [consistent](https://en.wikipedia.org/wiki/Consistent_heuristic)) for A* to be optimal.
In other words:
Optimal if h() is admissible, with tree search (no elimination repeated nodes)
Optimal if h() is consistent, with graph search (elimination repeated nodes)







Factoring:
A v A = A







Is [a] logically entailed by the KB? Explain why. 
Formula [a] is logically entailed by the KB because resolution is a sound inference procedure







tree policy: method used to choose the most urgent node to visit

default policy: policy of choosing the nodes inside a simulation







(4) Arc consistency algorithm cannot be applied to the problem formulated in (1) because the constraints are not binary, and thus a constraint graph cannot be defined.







# Montecarlo Tree Search

Problem: the tree is deep in the search space: the time in order to find a solution would be exponential.
What tries to do montecarlo? tries to find some approximate solutions to the problem.
Let's try to solve the game of chess. it is very large and so far there is no solution to it right now, because it is too large.
The problem of cess is that the payoff are avaialbe only at the end fo the game so you need to finish the martch in order tounderstand what is the outcome, but you have to compute a big number of matches and there is not enough time. wht is the typucal way of reducing the complexity? limiting the depth. let's fix 10 as height limit, but in the case of chess, 10 moves ahead are not enough surely. You have an agent and it needs to make a move, what he can try to do?, in this tree, where the payoffs are atvilable only at the end i stop the swarch at some level and i put some fictitious payoffs at that level.
At the beginning you have the initial situation, you start building the tree and after a while you stop. when you stop you have to put the payoff, but obviously you don't have it.so you ask yourself is it a good state or a bad state? 



the e valuation of the such state will be obtained by simulating the game starting from that state and than for instance playing randomly. you reach a final state and you take the value of this Montecarlo simulation with two random players and you put it to that bad/good state we were talking about.
The possible outcomes of this path are a lot obviously, are a distribution. if you repeat multiple times the simulation you will get different results. you will take the average of these results as an indicator of "how good is this state?"

The assumption is that we can simulate this part of the game very quickly, otherwise it's not efficient.

What is the problem of the average of the result computed?
the result of this random evaluation should be computed 
avoiding to build the tree when i see there are some states where i will lose with very high possiblity.
-->we need a kind of heuristic!
I want to stay toward the states that are better for me
Exploaration: try alternatives
Exploitation: go toward the explorations that are more promising.
We need to balance exploration and exploitation
The key factor to balance is what is called Optimist F. Uncertainty (OFU):
If you are very uncertain about something this uncertainty needs optimisim. you you are very uncertain about the performance of a state give him a bonus, when you get more and more confident you reduce the bonus.

So here we are, we have two values for each node.

- the result of the montecarlo simulation: mu hat.
  the average of the outcomes of the montecarlo simulations.
  (if you use just this criteria you are exploiting, not exploring)
- An upperbound that is a value that tells how much this estimate is uncertain



Algorithm:
First, what are we looking for?
We just want to solve a planning problem. Giving a state we want to understand which action to take.
Ergo, consider any state of the tree, for that state we want to answer what action to perform.

Node information:
N = number of Montecarlo simulation starting from this node.
Q = average of the results of the simulations starting from such node.

4 steps:

1. Selection

   - As long as the root is not fully expanded you keep on expanding the root.

     otherwise:
     you have to select a node using the following formula:
     ![1549987628116](C:\Users\Willi\AppData\Roaming\Typora\typora-user-images\1549987628116.png)
     Consider the node with the maximum value of U.
     if such node is not fully expanded select it
     otherwise: 
     compute the upperbound for its children and select the one children with  the highest upperbound. If there is a tie then select the left-most one, or the right-most if you are weird (it's up to you, do whatever you want as long as it is not specified).

     

2. Expansion

   - expand the child selected and initialize its Q and N to zero.

3. Simulation

   - Make up a random result for such child (win,lose, tie).

4. Update or Backup

   - Update all Qs and Ns of the subject node and his ancestors.

this algorithm is Any Time: we can repeat these steps as long as we want and then stop.










# PLANNING

### Closed World Assumption

The Closed World Assumption amounts to consider as false all the sentences that are not known to be true. 
In STRIPS, this means that all the predicates not listed in the representation of a state are considered false. 

### Terminology

- *<u>Predicate</u>*

  - something that can be true or false. stuff with parameters or without. as parameters, parameters take constants.
    a predicate with more than one parameter is called relation

  - Predicates can be divided in two classes:
    - *Fluent*
      Predicates that can change with time (true in some instances of time, false in others)
      (e.g. onTable(A))
      They are the only thing that change!
    - *Atemporal predicate*
      (e.g. Ball(A))

- *<u>Costants</u>*
  A, B : denotes object

- *<u>Primitives & Derivables</u>*

  Clear(A) is derivable because a block is clear if there is not a block upon it.

### State

A state is represented by a set of literals that are:

- *positive*
- *grounded* 
  they don't have any variable
- *function free*
  there are no functions



### Goal

- PDLL

  - Set of literals that are function-free
    e.g. 
    not On(A,B) 
    On(x,A)
    can be negative and can contain variables!

  - PDLL -> extends STRIPS
    
  - if you have a variable in a goal than this variable has an existence quantifier
    On(x,A) means Exists x |on(x,A) is true?

  - Goals are a set of states 
    [ C/A/B ]  or [ A/B   C ]     --> both satisfy on(A,B)

  - A state S satisfies a goal G when the state S contains all the positive literals of G and does not contain any of the negative literals of G
    

- STRIPS 

  - set of positive literals without variables
    STRIPS doesn't allow negative goals and variables in goal

    

### Action Schemas & Actions

Valid for both STRIPS and PDLL:

- Action schema:
  a set of possible actions, an action is derived by an action schema according to how I instantiate the variables.
  divided in

  - *<u>Name</u>*

  - *<u>Preconditions</u>*
    list of literals that are function free that state what should be true in the current state in order for the action to be applicable.
    definition:

    - precondition allows to decide if an action is applicable in a state.
      An action is applicable in a state when all the positive preconditions of the action are present in the description of the state
      (and not any negative precondition of the action is present in the description of the state)
      In fact he precondition can also contain some negative literals (something that Amigoni doesn't like). 

  - *<u>Effects</u>*

    1. copy the representation of the previous state
    2. delete the negative effects.
    3. add the positive effects.

    do not mention atemporal predicates in the actions please.

***frame problem***
transitioning from a state to the other most of the things don't change. (I wrote it just because he could ask it at the exam).
Example? disney cartoons LoL. fixed background, mickey mouse moves just moves his legs and arms e.e



### Forward Planning / Progressive Planning

the initial state 



### Backward Planning / Regression Planning

let's differentiate between two possible actions types:

- relevant and consistent actions: 
  actions that achieve a subgoal
- relevant but not consistent actions: 
  actions that achieve a subgoal but negate another subgoal

given an action A and a goal G, such that A is relevant and consistent for the goal G, the regression of the goal G through the action A is the goal G'
R[G,A] = G'

if we have a state S that satisfies G' it means that we can apply the action A that reaches a state A that satisfies G.

Backward Planning searchc in the space of goals

In Practice:

g' is found by copying g, delete positive effects of the action, adding all the preconditions of A



some goals g' will not be consistent , I would  need a consistency check but usually it's not done. 
Depth first search would suck! limited depth search would be ok, other searchs as well.

*implementation:*

1. the initial state is the goal
2. build all the g' from g
3. find a goal that is satisfied by the initial state



### Hierarchical Task Network

- Search in the space of plans, which means: let's start from an empty plan (just initial state + goal state) as the root. its children will be all the plans with only one action going from the initial state to the goal state. Their children will have two actions and so on until we find a plan that actually is feasible for reaching the goal state.
- An optimization consists in the Partial Ordering Planning which constraints the actions of the plan to respect a certain order



### Situation Calculus

- Start from a planning problem and transform it into a satisfiability problem, which means in a very big propositional logic formula.

- a situation is a picture of the world

- situations are objects, 

- reification: give names to objects

- at(robot1,room6,s_3)   it's a fluent --> it is true for situation 3 but it can be false for s_4

- by convention the situation is the last argument of the thing.

- so, you have logical formulas that are changing their truth values in time.

- *<u>Result Action</u>*
  we define very special elements, one of this elements is a function that is called "Result"
  Result takes an action and a situation and returns a new situation:
  Result( Movesnorth(Product), S_1) = S_2
  A situation can be thought of as a state.
- it's boring to express everything all the time. 
  we don't have the closed world assumption! 

$$
\forall x \forall s  \space Present(x,s) \space \and \space Portable(x) \rightarrow \space Holding(x,Result(Grab(x),s))
$$



- some predicates are fluent and some are not. 
  Portable(x) is not fluent, it's always portable or not.
  the fact that I'm holding an object is fluent. Present as well. 

- The preconditions are on the left side!
  The effect is on the right side!
  the name of the action is on the right side as well.
  left side: it's called the *<u>effect axiom</u>* : what is the effect of he action I'm performing

- every time that I have an *x* and a *s* such that the precondition is true and I'm performing the action of grabbing x then it is true. 

- another example of another effect axiom
  $$
  \forall x \forall s \space \neg Holding(x,Result(Drop(x),s))
  $$
  

  In this case no precondition.
  it says that if I'm dropping x in a situation s then i will reach a situation in which im not holding x.

- <u>*frame axiom</u>*

  Unfortunately we have to specify even what is not changing
  it consists in describing what is not changing:

$$
\forall x \forall c\forall s \space Color(x,c,s)\rightarrow Color(x,c,Result(Grab(x),s))If I grab an object x, its color c doesn't change.
$$

​	If I grab an object x, its color c doesn't change.


​	Another way of writing it is with functions (this means that we can use functions in situation calculus)
$$
\forall x\forall s \space Color(x,s)=Color(x,Result(Grab(x),s))
$$




- *In synthesis for each action that you have you should define an effect axiom + a number of frame axioms. moreover you need to specify an initial state (a set of conjuncted predicates) and goal state. how is the goal specified? it is something like this: it's usually an existentially quantified formula*

$$
\exists x \exists s \space Holding(x,s)
$$

​	(don't know if I copied it completely correctly, maybe the exists s has not been written by Amigoni)

​	can I derive this theorem above from the KB? Planning to proving a theorem
​	kb |- alpha where alpha is the thing above.



- STRIPS WAS CREATED TO SIMPLIFY ALL THIS MESS.
  but:
  SITUATION CALCULUS --> TOO COMPLEX --> MOVE TO STRIPS --> SOMETIMES DIFFUCULT --> MOVE TO SITUATION CALCLULUS BUT USE PROPOSITIONAL LOGIC INSTEAD OF FIRST ORDER LOGIC.











