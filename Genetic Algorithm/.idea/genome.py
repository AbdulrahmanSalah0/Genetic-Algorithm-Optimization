import time
from collections import namedtuple
from random import choices,randint,randrange,random
from typing import Callable,Tuple,List, Optional
from functools import partial

Genome = List[int]
Population = List[Genome]
FitnessFunc = Callable[[Genome],int]
PopulateFunc =Callable[[],Population]
SelectionFunc = Callable[[Population,FitnessFunc],Tuple[Genome,Genome]]
CrossoverFunc = Callable[[Genome,Genome],Tuple[Genome,Genome]]
MutationFunc = Callable[[Genome],Genome]
Thing = namedtuple('Thing',['name','value','weight'])

things = [
    Thing('Laptop',500,2200),
    Thing('headphone',150,160),
    Thing('coffee mug',60,350),
    Thing('Notepad',40,333),
    Thing('water Bottle',30,192),
]
more_things = [
                  Thing('mints',5,25),
                  Thing('socks',10,38),
                  Thing('tissus',15,80),
                  Thing('phone',500,200),
                  Thing('baseball cap',100,70),
              ]+ things
def generate_genome(length : int ) -> Genome :
    return choices([0,1], k = length )


def generate_Population(size: int , genome_length : int )-> Population :
    return [generate_genome(genome_length) for _ in range(size)]

def fitness(genome : Genome , things : [Thing] , weight_limit : int) -> int:
    if len(genome) != len(things):
        raise ValueError("genome and things must be the same lenght")
    weight = 0
    value = 0
    for i, thing in enumerate(things) :
        if genome[i] == 1:
            weight += thing.weight
            value += thing.value
            if weight > weight_limit :
                return 0
    return value
def selection_pair(population: Population,fitness_func :FitnessFunc) ->Population:
    return choices(
        population =population,
        weights=[fitness_func(genome)for genome in population],
        k =2
    )
def single_point_crossover(a : Genome,b : Genome) -> Tuple[Genome ,Genome]:
    if len(a) != len(b):
        raise ValueError("Genomes a and b must be of the same length")
    length = len(a)
    if length <2 :
        return a,b
    p = randint(1,length -1)
    return a[0:p] + b[p:] ,b[0:p]+a[p:]
def mutation(genome : Genome,num: int = 1,probability : float = 0.5) -> Genome:
    for _ in range(num) :
        index = randrange(len(genome))
        genome[index]=genome[index]if random() > probability else abs(genome[index-1])
    return genome
def run_evalution(
        populate_func: PopulateFunc ,
        fitness_func : FitnessFunc,
        fitness_limit : int ,
        selection_func :SelectionFunc = selection_pair,
        crossover_func : CrossoverFunc =single_point_crossover,
        mutation_func : MutationFunc = mutation ,
        generation_limit : int =100 ,
) -> Tuple[Population,int] :
    population = populate_func()

    for i in range(generation_limit):
        population = sorted(population,key=lambda genome: fitness_func(genome), reverse=True
                            )

        if fitness_func(population[0]) >= fitness_limit:
            break
        next_generation = population[0:2]

        for j in range(int(len(population) /2 -1)):
            parents = selection_func(population,fitness_func)
            offspring_a ,offspring_b =crossover_func(parents[0],parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation+=[offspring_a,offspring_b]
    population = sorted(
        population,key=lambda genome: fitness_func(genome),
        reverse=True
    )
    return population, i
start = time.time()
population ,generations = run_evalution(
    populate_func= partial(
        generate_Population,size = 10,genome_length =len(more_things)
    ),
    fitness_func=partial(
        fitness,things=more_things,weight_limit=3000
    ),
    fitness_limit=1310,
    generation_limit=100,

)
end = time.time()
def genome_to_things(genome: Genome,things:[Thing])->[Thing]:
    result = []
    for i , thing in enumerate(things):
        if genome[i] ==1 :
            result += [thing.name]
    return result
print(f"number of generations :{generations}")
print(f"time :{end - start}s")
print(f"best soultion  :{genome_to_things(population[0],more_things)}")