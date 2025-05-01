use std::collections::HashMap;

#[derive(Default)]
pub struct UnionFind {
    parents: Vec<usize>,
    ranks: Vec<usize>,
    sizes: Vec<usize>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        UnionFind {
            parents: (0..n).collect(),
            ranks: vec![0; n],
            sizes: vec![1; n],
        }
    }

    pub fn representative(&mut self, x: usize) -> usize {
        if self.parents[x] == x {
            return x;
        }
        let p = self.representative(self.parents[x]);
        self.parents[x] = p;
        p
    }

    pub fn merge(&mut self, x: usize, y: usize) {
        let x = self.representative(x);
        let y = self.representative(y);
        if x == y {
            return;
        }

        let (x, y) = if self.ranks[x] < self.ranks[y] {
            (y, x)
        } else {
            (x, y)
        };
        self.parents[y] = x;
        self.sizes[x] += self.sizes[y];
        if self.ranks[x] == self.ranks[y] {
            self.ranks[x] += 1;
        }
    }

    pub fn is_same(&mut self, x: usize, y: usize) -> bool {
        self.representative(x) == self.representative(y)
    }

    pub fn size(&mut self, x: usize) -> usize {
        let x = self.representative(x);
        self.sizes[x]
    }

    pub fn append(&mut self, n: usize) {
        let m = self.parents.len();
        self.parents.extend((m..m + n).collect::<Vec<_>>());
        self.ranks.extend(vec![0; n]);
    }

    pub fn groups(&mut self) -> Vec<Vec<usize>> {
        let mut groups = HashMap::new();
        for i in 0..self.parents.len() {
            let repr = self.representative(i);
            groups.entry(repr).or_insert_with(Vec::new).push(i);
        }
        groups.into_values().collect()
    }
}

#[cfg(test)]
pub mod test {
    use super::*;

    impl UnionFind {
        pub fn assert(&mut self, lhs: Vec<Vec<usize>>) {
            let mut rhs = self.groups();
            for e in rhs.iter_mut() {
                e.sort();
            }
            rhs.sort();
            assert_eq!(lhs, rhs);
        }
    }

    #[test]
    fn test_union_find() {
        let mut uf = UnionFind::new(6);

        // [0], [1], [2], [3], [4], [5]
        uf.assert(vec![vec![0], vec![1], vec![2], vec![3], vec![4], vec![5]]);

        uf.merge(0, 1);
        uf.merge(2, 3);
        uf.merge(1, 5);

        // [0, 1, 5], [2, 3], [4]
        uf.assert(vec![vec![0, 1, 5], vec![2, 3], vec![4]]);
    }
}
