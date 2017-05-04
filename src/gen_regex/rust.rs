use regex::Regex;
use regex_syntax::Expr;
//use regex_syntax::Expr::*;
use regex_syntax::Repeater;
//use itertools::*;

#[derive(Copy,Clone,Debug,Eq,PartialEq)]
pub enum Epsilon {
    WordBoundary,
    NonWordBoundary,
    Terminus,
    None,
    WordTerminus,
    NonWordTerminus
}

fn crosses_boundary(a:&String,b:&String) -> bool {
    (end_unword(a) && start_word(b)) || (end_word(a) && start_unword(b))
}

fn end_unword(a:&String) -> bool {
    lazy_static! {
        static ref ENDUNWORD: Regex = Regex::new("(\\W|\\A|\\z)$").unwrap();
    }
    ENDUNWORD.is_match(a)
}

fn start_unword(a:&String) -> bool {
    lazy_static! {
        static ref STARTUNWORD: Regex = Regex::new("^(\\W|\\A|\\z)").unwrap();
    }
    STARTUNWORD.is_match(a)
}

fn start_word(a:&String) -> bool {
    lazy_static!{
        static ref STARTWORD: Regex = Regex::new("^\\w").unwrap();
    }
    STARTWORD.is_match(a)
}

fn end_word(a:&String) -> bool {
    lazy_static! {
        static ref ENDWORD: Regex = Regex::new("\\w$").unwrap();
    }
    ENDWORD.is_match(a)
}

impl Epsilon {
    pub fn merge(&self,other:&Self) -> Option<Self> {
        match (*self,*other) {
            (Epsilon::None,x) => Some(x),

            (Epsilon::WordBoundary, Epsilon::WordBoundary) => Some(Epsilon::WordBoundary),
            (Epsilon::WordBoundary, Epsilon::NonWordBoundary) => None,
            (Epsilon::WordBoundary, Epsilon::Terminus) => Some(Epsilon::WordTerminus),
            (Epsilon::WordBoundary, Epsilon::WordTerminus) => Some(Epsilon::WordTerminus),
            (Epsilon::WordBoundary, Epsilon::NonWordTerminus) => None,

            (Epsilon::NonWordBoundary,Epsilon::NonWordBoundary) => Some(Epsilon::NonWordBoundary),
            (Epsilon::NonWordBoundary,Epsilon::Terminus) => Some(Epsilon::NonWordTerminus),
            (Epsilon::NonWordBoundary,Epsilon::WordTerminus) => None,
            (Epsilon::NonWordBoundary,Epsilon::NonWordTerminus) => Some(Epsilon::NonWordTerminus),

            (Epsilon::Terminus,Epsilon::Terminus) => Some(Epsilon::Terminus),
            (Epsilon::Terminus,Epsilon::WordTerminus) => Some(Epsilon::WordTerminus),
            (Epsilon::Terminus,Epsilon::NonWordTerminus) => Some(Epsilon::NonWordTerminus),

            (Epsilon::WordTerminus,Epsilon::WordTerminus) => Some(Epsilon::WordTerminus),
            (Epsilon::WordTerminus,Epsilon::NonWordTerminus) => None,

            (Epsilon::NonWordTerminus,Epsilon::NonWordTerminus) => Some(Epsilon::NonWordTerminus),
            _ => other.merge(&self)
        }
    }

    pub fn pass_as_mid(&self,left:&String,right:&String) -> bool {
        match *self {
            Epsilon::WordBoundary => !crosses_boundary(&left,&right),
            Epsilon::NonWordBoundary => crosses_boundary(&left,&right),
            Epsilon::WordTerminus => false,
            Epsilon::NonWordTerminus => false,
            Epsilon::Terminus => false,
            Epsilon::None => true
        }
    }

}
use std::fmt::Debug;
#[derive(Debug,Clone,PartialEq,Eq)]
pub struct SearchNode<T> where T:PartialEq + Clone + Eq + Debug{
    weight:usize,
    pub node:Option<T>
}

impl<T> SearchNode<T> where T:PartialEq + Clone + Eq + Debug{
    pub fn weight(&self) -> usize {
        self.weight
    }
}


impl SearchNode<EpString> {

    pub fn merge(self,other:EpString) -> Self {
        let new_weight = self.weight + other.len();
        if let Some(ep) = self.node {
            ep.merge(other)
        } else {
            SearchNode{weight:new_weight, node:None}
        }
    }
}

impl<T> Ord for SearchNode<T> where T:PartialEq + Clone + Eq + Debug{
    fn cmp(&self, other: &Self) -> Ordering {
    // Notice that the we flip the ordering here
    other.weight.cmp(&self.weight)
    }
}

// `PartialOrd` needs to be implemented as well.
impl<T> PartialOrd for SearchNode<T> where T:PartialEq + Clone + Eq + Debug{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}



#[derive(PartialEq,Eq,Clone,Debug)]
pub struct EpString {
    pre: Epsilon,
    pub s: Option<String>,
    post: Epsilon
}

impl EpString {
    pub fn raw_str(s: String) -> EpString {
        if s.len() > 0 {
            EpString{pre:Epsilon::None,s:Some(s),post:Epsilon::None}
        } else {
            EpString{pre:Epsilon::None,s:None,post:Epsilon::None}
        }
    }

    pub fn empty() ->EpString {
        EpString{pre:Epsilon::None,s:None,post:Epsilon::None}
    }

    pub fn search_wrap(self) -> SearchNode<EpString> {
        let weight = self.len();
        SearchNode{weight:weight,node: Some(self)}
    }

    pub fn fork_string(&self,new_s:String) -> EpString {
        EpString{pre:self.pre.clone(),s:Some(new_s),post:self.post.clone()}
    }

    pub fn epsilon(e:Epsilon) -> EpString {
        EpString{pre:e.clone(), s:None,post:e}
    }

    pub fn left_epsilon(e:Epsilon) -> EpString {
        EpString{pre:e, s:None,post:Epsilon::None}
    }

    pub fn right_epsilon(e:Epsilon) -> EpString {
        EpString{pre:Epsilon::None, s:None,post:e}
    }

    pub fn contradict(ghost_length:usize) -> SearchNode<EpString> {
        SearchNode{weight:ghost_length,node:None}
    }

    pub fn len(&self) -> usize {
        if let &Some(ref s) = &self.s {
            s.len()
        } else {
            0
        }
    }

    pub fn merge(self,other:Self) -> SearchNode<Self> {
        let ghost_len = self.len() + other.len();
        match (self.s,&other.s) {
            (Some(ref l),&Some(ref r)) => {
                if let Some(me) = self.post.merge(&other.pre) {
                    if !me.pass_as_mid(&l,&r) {
                        EpString::contradict(ghost_len)
                    } else {
                        let merged = l.clone() + &r;
                        EpString{pre:self.pre,s:Some(merged), post:other.post}.search_wrap()
                    }
                } else { EpString::contradict(ghost_len) }
            },
            (Some(l),&None) => {
                if let Some(t) = other.pre.merge(&other.post) {
                    if let Some(new_post) = t.merge(&self.post) {
                        EpString{pre:self.pre,s:Some(l),post:new_post}.search_wrap()
                    } else { EpString::contradict(ghost_len) }
                } else { EpString::contradict(ghost_len) }
            },
            (None,&Some(ref r)) => {
                if let Some(t) = self.pre.merge(&self.post) {
                    if let Some(new_post) = t.merge(&other.pre) {
                        EpString{pre:new_post,s:Some(r.clone()),post:other.post.clone()}.search_wrap()
                    } else { EpString::contradict(ghost_len) }
                } else { EpString::contradict(ghost_len) }
            },
            (None,&None) => {
                if let Some(_) = self.post.merge(&other.pre) {
                    EpString{pre:self.pre,s:None,post:other.post}.search_wrap()
                } else {
                    EpString::contradict(ghost_len)
                }
            }
        }
    }
}

use std::cmp::{Ord,Ordering};



impl PartialOrd for EpString {
    fn partial_cmp(&self,other:&Self) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

impl Ord for EpString {
    fn cmp(&self,other:&Self) -> Ordering {
        match (&self.s,&other.s) {
            (&Some(ref l),&Some(ref r)) => l.len().cmp(&r.len()),
            (&Some(_),&None) => Ordering::Greater,
            (&None,&Some(_)) => Ordering::Less,
            (&None,&None) => Ordering::Equal
        }
    }
}


pub struct Bits {
    curr: usize
}

impl Bits {
    fn new(t:usize) -> Bits {
        Bits{curr:t}
    }
}

impl Iterator for Bits {
    type Item = bool;
    fn next(&mut self) -> Option<bool> {
        if self.curr & 1 == 1 {
            self.curr >>=1;
            Some(true)
        } else {
            self.curr >>=1;
            Some(false)
        }
    }
}

pub struct BaseCaseIterator {
    case: SearchNode<EpString>,
    sent: bool
}

impl BaseCaseIterator {
    fn new(base:EpString) -> BaseCaseIterator {
        BaseCaseIterator{case:base.search_wrap(),sent:false}
    }
}

impl Iterator for BaseCaseIterator {
    type Item=SearchNode<EpString>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.sent {
            None
        } else {
            self.sent = true;
            Some(self.case.clone())
        }
    }
}

pub struct CaseInsensitiveIterator<I> {
    sub: Box<I>,
    curr: SearchNode<EpString>,
    curr_offset: usize,
    curr_max: usize
}

impl<I> CaseInsensitiveIterator<I> where I: Iterator<Item=SearchNode<EpString>>{
    pub fn new(mut it: I) -> CaseInsensitiveIterator<I> {
        if let Some(x) = it.next() {
            let mut t = CaseInsensitiveIterator{ sub: Box::new(it), curr: x, curr_offset: 0, curr_max: 0};
            t.reset();
            t
        } else {
            CaseInsensitiveIterator{ sub: Box::new(it), curr: EpString::epsilon(Epsilon::None).search_wrap(), curr_offset: 0, curr_max: 0}
        }
    }

    pub fn reset(&mut self) {
        let ll = self.curr.weight();
        if ll == 0 || self.curr.node == None { // An epsilon or a Skip
            self.curr_max = 1;
            self.curr_offset = 0;
        } else {
            self.curr_offset = 0;
            if 1 << ll < usize::max_value() {
                self.curr_max = 1 << ll;
            } else {
                self.curr_max = usize::max_value() - 1;
            }
        }
    }

    pub fn capital_mask(&self,mask:usize,st:String) -> String {
        st.chars()
            .zip(Bits::new(mask))
            .map(|(c,b)| {
                if b {
                    c.to_uppercase().collect::<String>()
                } else {
                    c.to_lowercase().collect::<String>()
                }})
            .collect()
    }

}

impl<I> Iterator for CaseInsensitiveIterator<I> where I: Iterator<Item=SearchNode<EpString>>{
    type Item=SearchNode<EpString>;
    fn next(&mut self) -> Option<Self::Item> {
        // Cases
        // 1: Exhausted, already sent last
        // 2: The current node is a Skip
        // 3: The current string is an Epsilon
        // 4: Another permutation of the current string
        // 5: The current string permutations have been exhausted, but there are more in the Iterator
        // 6: Exhausted, and nothing left

        if self.curr_max == 0 {
            None // Case 1
        } else if self.curr_max == 1 {
            self.curr_max = 0; // Case 2 and 3 are the same case
            Some(self.curr.clone())
        } else if self.curr_offset < self.curr_max {
            if let &Some(ref ep) = &self.curr.node {
                if let Some(ref s) = ep.s {
                    let m = self.curr_offset;
                    self.curr_offset += 1;
                    Some(ep.fork_string(self.capital_mask(m,s.clone())).search_wrap())
                } else {
                    panic!("Case Insensitive - Epsilon uncaught");
                }
            } else {
                panic!("Case Insensitive - Skip Node uncaught");
            }
        } else {
            if let Some(x) = self.sub.next() {
                // Case 4
                self.curr = x;
                self.reset();
                self.next()
            } else {
                // Case 5
                self.curr_max = 0;
                None
            }
        }
    }
}




// pub trait REGenerate{
//     fn gen(&self) -> Iterator<Item=EStringE>;
// }
pub struct AnyChar {
    nl: bool,
    ind: u32
}

impl AnyChar {
    pub fn new(nl:bool) -> AnyChar {
        AnyChar{nl:nl,ind:1}
    }
}
use std::char::from_u32;

impl Iterator for AnyChar {
    type Item = SearchNode<EpString>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.ind > 17825791 {
            None
        } else {
            while !self.nl &&
                self.ind == 0xA ||
                self.ind == 0xB ||
                self.ind == 0xC ||
                self.ind == 0xD ||
                self.ind == 0x85 ||
                self.ind == 0x2028 ||
                self.ind == 0x2029
            {
                self.ind += 1;
            }
            if let Some(r) = from_u32(self.ind) {
                self.ind += 1;
                let mut ret = String::new();
                ret.push(r);
                Some(EpString::raw_str(ret).search_wrap())
            } else {None}
        }
    }
}

use regex_syntax::ClassRange;

pub struct CharStreamWrapper<I>{
    it:I,
    curr_index: u32,
    max_index: u32
}

impl<I> CharStreamWrapper<I> where I : Iterator<Item=ClassRange> {
    pub fn new(mut it: I) -> CharStreamWrapper<I> {
        let (curr,max_i) = if let Some(r) = it.next() {
            (r.start as u32, r.end as u32)
        } else {
            (0,0)
        };
        CharStreamWrapper{it:it,curr_index:curr, max_index:max_i}
    }
}

impl<I> Iterator for CharStreamWrapper<I> where I : Iterator<Item=ClassRange>{
    type Item = SearchNode<EpString>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr_index <= self.max_index {
            let mut s = String::new();
            s.push(from_u32(self.curr_index).unwrap());
            self.curr_index += 1;
            Some(EpString::raw_str(s).search_wrap())
        } else if self.max_index == 0 {
            None
        } else {
            if let Some(r) = self.it.next() {
                self.curr_index = r.start as u32;
                self.max_index = r.end as u32;
                self.next()
            } else {
                None
            }
        }
    }
}

use std::collections::BinaryHeap;

pub struct HTStream<T> where T: Clone + PartialEq + Ord + PartialOrd + Debug {
    tail: Box<Iterator<Item=SearchNode<T>>>,
    head: Option<SearchNode<T>>
}

impl<T> HTStream<T> where T: Clone + PartialEq + Ord + PartialOrd + Debug {
    pub fn new(tail: Box<Iterator<Item=SearchNode<T>>>) -> HTStream<T>{
        let mut temp_tail = tail;
        let head = temp_tail.next();
        HTStream{head:head,tail:temp_tail}
    }

    pub fn exhausted(&self) -> bool {
        self.head == None
    }
}

impl<T> Ord for HTStream<T> where T: Clone + PartialEq + Ord + PartialOrd + Debug {
    fn cmp(&self,other:&Self) -> Ordering {
        self.head.cmp(&other.head)
    }
}

impl<T> PartialOrd for HTStream<T> where T: Clone + PartialEq + Ord + PartialOrd + Debug {
    fn partial_cmp(&self,other:&Self) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

impl<T> PartialEq for HTStream<T> where T: Clone + PartialEq + Ord + PartialOrd + Debug {
    fn eq(&self, other: &Self) -> bool {
        self.head == other.head
    }
}

impl<T> Eq for HTStream<T> where T: Clone + PartialEq + Ord + PartialOrd + Debug {}

use std::mem::swap;

impl<T> Iterator for HTStream<T>  where T : PartialEq + Ord + PartialOrd + Clone + Debug {
    type Item = SearchNode<T>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.head == None {
            None
        } else {
            let last_weight = if let &Some(ref r) = &self.head {r.weight} else {0};
            let mut nn = None;
            for t in &mut self.tail {
                if let Some(_) = t.node {
                    nn = Some(t);
                    break;
                } else {
                    if t.weight > last_weight {
                        nn = Some(t);
                        break;
                    }
                }
            }
            swap(&mut self.head,& mut nn);
            nn
        }
    }
}


#[derive(Debug,Eq,PartialEq,Clone)]
pub struct Repeatable {
    s: EpString,
    rep_count: u32,
    weight: usize,
    ind: usize
}

impl Ord for Repeatable {
    fn cmp(&self,other:&Self) -> Ordering {
        if self.weight == other.weight {
            self.ind.cmp(&other.ind)
        } else {
            other.weight.cmp(&self.weight)
        }
    }
}

impl PartialOrd for Repeatable {
    fn partial_cmp(&self,other:&Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct RepeatStream {
    frontier: BinaryHeap<ConcatIdx>,
    seen: HashSet<ConcatIdx>,
    stream: MaterializedStream<EpString>,
    max: Option<u32>
}

impl RepeatStream {
    pub fn new(r:Repeater, it:Box<Iterator<Item=SearchNode<EpString>>>) -> RepeatStream {
        let (min,max) = match r {
            Repeater::ZeroOrOne => (0,Some(1)),
            Repeater::ZeroOrMore => (0,None),
            Repeater::OneOrMore => (1,None),
            Repeater::Range{min,max} => (min,max)
        };
        let mut idx_vec = Vec::new();
        for _ in 0..min {
            idx_vec.push(0);
        }
        let mut heap = BinaryHeap::new();
        heap.push(ConcatIdx{weight:0,idx:idx_vec});
        let mut ret = RepeatStream{frontier:heap, seen:HashSet::new(),stream:MaterializedStream::new(it), max:max};
        ret.stream.force_pull();
        ret
    }

    fn get_cost(&self, idx:&Vec<usize>) -> Option<usize>{
        let mut cost = Some(0);
        for i in idx.iter() {
            cost = match (cost,self.stream.cost(i.clone())) {
                (Some(x),Some(y)) => Some(x+y),
                _ => None
            }
        }
        cost
    }

    fn materialized_idx(&self, idx:&ConcatIdx) -> bool {
        let max_idx = idx.idx.iter().fold(0,|acc,n| if &acc > n {acc} else {n.clone()});
        self.stream.materialized_idx(max_idx)
    }

    fn have_seen(&self, idx:&ConcatIdx) -> bool {
        self.seen.contains(idx)
    }

    fn see(&mut self, idx:ConcatIdx) {
        self.seen.insert(idx);
    }

    fn plan_novel(&mut self, idx:ConcatIdx) {
        if !self.have_seen(&idx) {
            self.see(idx.clone());
            self.frontier.push(idx)
        }
    }

    fn pull_idx(&mut self, idx:&ConcatIdx) {
        let max_idx = idx.idx.iter().fold(0,|acc,n| if &acc > n {acc} else {n.clone()});
        while !self.stream.materialized_idx(max_idx) {
            self.stream.pull();
        }
    }

    fn plan_succesors(&mut self,idx:ConcatIdx) {
        if self.materialized_idx(&idx) {
            for mut new_idx in (0..idx.len()).map(|i| idx.succesor(i)) {
                if let Some(cost) = self.get_cost(&new_idx.idx) {
                    new_idx.weight = cost;
                    self.pull_idx(&new_idx);
                    self.plan_novel(new_idx);
                }
            }
            match (self.max,idx.len()) {
                (Some(max), x) if x == max as usize => {},
                _ => {
                    let mut appended = idx;
                    appended.idx.push(0);
                    appended.weight = self.get_cost(&appended.idx).unwrap();
                    self.plan_novel(appended);
                }
            }

        } else {
            self.pull_idx(&idx);
            if let Some(cost) = self.get_cost(&idx.idx) {
                let mut new_idx = idx;
                new_idx.weight = cost;
                self.plan_novel(new_idx);
            } else {
            }
        }
    }

    fn output(&self,idx:&ConcatIdx) -> SearchNode<EpString> {
        (idx.idx.iter().map(|n| self.stream[n.clone()].clone()).fold(EpString::empty().search_wrap(),|acc,s| acc.merge(s)))
    }


    fn pop(&mut self) -> Option<ConcatIdx> {
        self.frontier.pop()
    }
}

impl Iterator for RepeatStream{
    type Item = SearchNode<EpString>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut ret = None;
        while ret == None {
            if let Some(idx) = self.pop() {
                ret = if self.materialized_idx(&idx) {
                    Some(self.output(&idx))
                } else {
                    None
                };
                self.plan_succesors(idx);
            } else {
                return None
            }
        }
        ret
    }
}

pub struct AlternateIterator {
    heap:BinaryHeap<HTStream<EpString>>
}

impl AlternateIterator {
    pub fn new(its: Vec<Box<Iterator<Item=SearchNode<EpString>>>>) -> AlternateIterator {
        let mut bh = BinaryHeap::new();
        for it in its {
            let nn = HTStream::new(Box::new(it));
            if !nn.exhausted() {
                bh.push(nn);
            }
        }
        AlternateIterator{heap:bh}
    }
}

impl Iterator for AlternateIterator {
    type Item = SearchNode<EpString>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut ret = None;

        while let Some(mut ht) = self.heap.pop() {
            if let Some(h) = ht.next() {
                if !ht.exhausted() {
                    self.heap.push(ht);
                }
                ret = Some(h);
                break
            } else {
                panic!("Alternator added to heap an exhausted iterator.");
            }
        }
        ret
    }
}

pub struct MaterializedStream<T> where T: PartialOrd + Ord + PartialEq + Clone + Debug {
    materialized: Vec<T>,
    costs: Vec<usize>,
    tail: HTStream<T>
}

impl<T> MaterializedStream<T> where T: PartialOrd + Ord + PartialEq + Clone + Debug {

    pub fn new(it:Box<Iterator<Item=SearchNode<T>>>) -> MaterializedStream<T> {
        let mut ret = MaterializedStream{materialized:Vec::new(),costs:Vec::new(),tail:HTStream::new(it)};
        while ret.materialized.len() < 1 && ret.pull(){
        }
        ret
    }

    pub fn materialized_idx(&self, idx:usize) -> bool {
        self.materialized.len() > idx
    }

    pub fn empty(&self) -> bool {
        self.materialized.len() == 0
    }

    pub fn cost(&self, idx:usize) -> Option<usize> {
        if idx < self.costs.len() {
            Some(self.costs[idx])
        } else if idx == self.costs.len() {
            if let Some(ref ht) = self.tail.head {
                Some(ht.weight)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn exhausted(&self) -> bool {
        self.tail.head == None
    }

    pub fn force_pull(&mut self) -> bool {
        for i in &mut self.tail {
            let weight = i.weight;
            if let Some(node) = i.node {
                self.materialized.push(node);
                self.costs.push(weight);
                return true
            }
        }
        false
    }

    pub fn pull(&mut self) -> bool {
        if self.exhausted() {
            return false
        }
        if let Some(tail_next) = self.tail.next() {
            if let Some(node) = tail_next.node {
                self.materialized.push(node);
                self.costs.push(tail_next.weight);
                true
            } else {
                true
            }
        } else {
            true
        }
    }
}


use std::collections::HashSet;

#[derive(PartialEq,Eq,Clone,Debug,Hash)]
pub struct ConcatIdx {
    weight:usize,
    idx:Vec<usize>
}

impl ConcatIdx {
    pub fn succesor(&self,idx:usize) -> ConcatIdx {
        let mut ret = self.clone();
        ret.idx[idx] += 1;
        ret
    }

    pub fn len(&self) -> usize {
        self.idx.len()
    }
}

impl Ord for ConcatIdx {
    fn cmp(&self, other:&Self) -> Ordering {
        other.weight.cmp(&self.weight)
    }
}

impl PartialOrd for ConcatIdx {
    fn partial_cmp(&self,other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct ConcatIterator {
    mat: Vec<MaterializedStream<EpString>>,
    frontier: BinaryHeap<ConcatIdx>,
    seen: HashSet<ConcatIdx>,
    seen_cap: usize
}

impl ConcatIterator {
    pub fn new(its:Vec<Box<Iterator<Item=SearchNode<EpString>>>>) -> ConcatIterator {
        let mut mat = Vec::new();
        let mut idx_vec = Vec::new();
        let mut scrap = false;
        for it in its {
            idx_vec.push(0);
            let ms = MaterializedStream::new(it);
            scrap = scrap | ms.empty();
            mat.push(ms);
        }
        let mut front = BinaryHeap::new();
        if !scrap {front.push(ConcatIdx{weight:0,idx:idx_vec});}
        ConcatIterator{
            mat:mat,
            frontier:front,
            seen:HashSet::new(),
            seen_cap:0
        }
    }

    fn get_cost(&self,idx:&Vec<usize>) -> Option<usize> {
        let mut cost = Some(0);
        for (i,n) in idx.iter().enumerate() {
            cost = match (cost,self.mat[i].cost(n.clone())) {
                (Some(x),Some(y)) => Some(x+y),
                _ => None
            }
        }
        cost
    }

    fn materialized_idx(&self, idx:&ConcatIdx) -> bool {
        idx.idx.iter().enumerate().map(|(i,n)| self.mat[i].materialized_idx(n.clone())).fold(true,|acc,r| acc && r)
    }

    fn have_seen(&self, idx:&ConcatIdx) -> bool {
        idx.weight < self.seen_cap || self.seen.contains(idx)
    }

    fn see(&mut self, idx:ConcatIdx) {
        self.seen.insert(idx);
    }

    fn plan_novel(&mut self, idx:ConcatIdx) {
        if !self.have_seen(&idx) {
            self.see(idx.clone());
            self.frontier.push(idx)
        }
    }

    fn get_frontier(&mut self) -> Option<ConcatIdx> {
        self.frontier.pop()
    }

    fn pull_idx(&mut self, idx:&ConcatIdx) {
        for (i,n) in idx.idx.iter().enumerate() {
            if !self.mat[i].materialized_idx(n.clone()) {
                self.mat[i].pull();
            }
        }
    }

    fn plan_succesors(&mut self,idx:ConcatIdx) {
        if self.materialized_idx(&idx) {
            for mut new_idx in (0..idx.len()).map(|i| idx.succesor(i)) {
                if let Some(cost) = self.get_cost(&new_idx.idx) {
                    new_idx.weight = cost;
                    self.pull_idx(&new_idx);
                    self.plan_novel(new_idx);
                }
            }
        } else {
            self.pull_idx(&idx);
            if let Some(cost) = self.get_cost(&idx.idx) {
                let mut new_idx = idx;
                new_idx.weight = cost;
                self.plan_novel(new_idx);
            } else {
            }
        }
    }

    fn output(&self,idx:&ConcatIdx) -> SearchNode<EpString> {
        (idx.idx.iter().enumerate().map(|(i,n)| self.mat[i][n.clone()].clone()).fold(EpString::empty().search_wrap(),|acc,s| acc.merge(s)))
    }
}

impl Iterator for ConcatIterator {
    type Item = SearchNode<EpString>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut ret = None;
        while ret == None {
            if let Some(idx) = self.get_frontier() {
                ret = if self.materialized_idx(&idx) {
                    Some(self.output(&idx))
                } else {
                    None
                };
                self.plan_succesors(idx);
            } else {
                return None
            }
        }
        ret
    }
}

use std::ops::Index;

impl<T> Index<usize> for MaterializedStream<T> where T: PartialOrd + Ord + PartialEq + Clone + Debug {
    type Output = T;
    fn index(&self,idx:usize) -> &Self::Output {
        &self.materialized[idx]
    }
}




pub fn re_gen(e: Expr) -> Box<Iterator<Item=SearchNode<EpString>>> {
    match e {
        Expr::Empty => Box::new(BaseCaseIterator::new(EpString::epsilon(Epsilon::None))),
        Expr::Literal{chars: c, casei: i} => {
            if i {
                Box::new(CaseInsensitiveIterator::new(BaseCaseIterator::new(EpString::raw_str(c.iter().collect::<String>()))))
            } else {
                Box::new(BaseCaseIterator::new(EpString::raw_str(c.iter().collect::<String>())))
            }
        },
        Expr::AnyCharNoNL => Box::new(AnyChar::new(false)),
        Expr::AnyChar => Box::new(AnyChar::new(true)),
        Expr::Class(x) => Box::new(CharStreamWrapper::new(x.into_iter())),

        Expr::StartLine => Box::new(BaseCaseIterator::new(EpString::left_epsilon(Epsilon::Terminus))),
        Expr::EndLine => Box::new(BaseCaseIterator::new(EpString::right_epsilon(Epsilon::Terminus))),
        Expr::StartText => Box::new(BaseCaseIterator::new(EpString::left_epsilon(Epsilon::Terminus))),
        Expr::EndText => Box::new(BaseCaseIterator::new(EpString::right_epsilon(Epsilon::Terminus))),
        Expr::WordBoundary => Box::new(BaseCaseIterator::new(EpString::epsilon(Epsilon::WordBoundary))),
        Expr::NotWordBoundary =>      Box::new(BaseCaseIterator::new(EpString::epsilon(Epsilon::NonWordBoundary))),
        Expr::WordBoundaryAscii =>    Box::new(BaseCaseIterator::new(EpString::epsilon(Epsilon::WordBoundary))),
        Expr::NotWordBoundaryAscii => Box::new(BaseCaseIterator::new(EpString::epsilon(Epsilon::NonWordBoundary))),

        Expr::Group{e, i:_, name:_} => re_gen(*e),
        Expr::Repeat{e, r, greedy:_} => Box::new(RepeatStream::new(r,re_gen(*e))),
        Expr::Concat(x) => Box::new(ConcatIterator::new(x.into_iter().map(|t| re_gen(t)).collect())),
        Expr::Alternate(x) => Box::new(AlternateIterator::new(x.into_iter().map(|t| re_gen(t)).collect())),
        _ => Box::new(vec![EpString::epsilon(Epsilon::None).search_wrap()].into_iter())

    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use regex_syntax::Expr;
    // use regex_syntax::properties;
    use time;

    #[test]
    fn double_z_one() {
        let re = "(?u:.)??";
        println!("-------{:?}--------", re);
        // let mut length = 0;
        let foo = Expr::parse(re);
        println!("{:?}",foo);
        // let test :Regex = Regex::new(re).unwrap();
        let mut cc = 0;
        let now = time::precise_time_ns();
        if let Ok(x) = foo {
            for sn in re_gen(x) {
                if let Some(ep) = sn.node {
                    if let Some(_) = ep.s {
                        cc += 1;
                        if cc % 10000 == 0 {println!("{}",time::precise_time_ns() - now); break}
                        //assert!(length <= s.len());
                        //assert!(test.is_match(&s));
                        //length = s.len();
                        //println!("{} - {:?}",i,s)
                    }
                }
            }
        }
        println!("--------------------");
    }

    #[test]
    fn class() {
        for re in ["((?u:[A-Za-zſ-ſK-K]){1,20})", ".", "a+","[ab]+"].into_iter() {
            println!("-------{:?}--------", re);
            let mut length = 0;
            let foo = Expr::parse(re);
            println!("{:?}",foo);
            // let test :Regex = Regex::new(re).unwrap();
            let mut cc = 0;
            let now = time::precise_time_ns();
            if let Ok(x) = foo {
                let mut search_count = 0;
                for sn in re_gen(x).take(1000) {
                    search_count += 1;
                    if let Some(ep) = sn.node {
                        if let Some(s) = ep.s {
                            cc += 1;
                            if cc % 10000 == 0 {println!("{}",time::precise_time_ns() - now); break}
                            assert!(length <= s.len());
                            //assert!(test.is_match(&s));
                            length = s.len();

                            //println!("{} - {:?}",i,s)
                        }
                    }
                }
            }
            println!("--------------------");
        }
    }

    use quickcheck::{QuickCheck,StdGen};

    #[test]
    fn print_err() {
        for re in ["(ab|c","ab)","[a", "]", "?"].into_iter() {
            let foo = Expr::parse(&re);
            println!("{:?} - {:?}",re,foo);
        }
    }

    use rand::{thread_rng,ThreadRng};

    #[test]
    fn gen_no_epsilons() {
        fn prop(e: Expr) -> bool {
            if let Ok(tester) = Regex::new(&e.to_string()) {
                println!("-----------------------------\n{:?}",&e.to_string());
                println!("{:?}", e);
                let mut length = 0;
                let test :Regex = Regex::new(&e.to_string()).unwrap();
                let mut cc = 0;
                let now = time::precise_time_ns();
                let mut search_count = 0;
                for sn in re_gen(e).take(1000) {
                    search_count += 1;
                    if let Some(ep) = sn.node {
                        if let Some(s) = ep.s {
                            cc += 1;
                            if cc % 10000 == 0 {println!("{}",time::precise_time_ns() - now); break}
                            assert!(length <= s.len());
                            if !test.is_match(&s) {
                                println!("Mismatch: {:?}", s);
                                assert!(false);
                            }
                            length = s.len();
                        }
                    }
                }
            }

            true
        }
        QuickCheck::new()
            .tests(100)
            .max_tests(200)
            .gen(StdGen::new(thread_rng(),30))
            .quickcheck(prop as fn(Expr) -> bool);
    }
}
