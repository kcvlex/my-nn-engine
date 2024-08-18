use cranelift::prelude::*;
use std::collections::{BTreeSet, HashMap};
use std::ops::Bound;

#[derive(PartialOrd, Ord, PartialEq, Eq, Clone, Debug)]
struct Region {
    start: u64,
    end: u64,
}

impl Region {
    fn size(&self) -> usize {
        (self.end - self.start) as usize
    }
}

struct Block {
    free: BTreeSet<Region>,
}

impl Block {
    #[cfg(test)]
    fn new(size: usize) -> Self {
        let region = Region {
            start: 0,
            end: size as u64,
        };
        let mut free = BTreeSet::new();
        free.insert(region);
        Block { free }
    }

    fn new_with_allocated(size: usize, allocated: usize) -> Self {
        let region = Region {
            start: allocated as u64,
            end: size as u64,
        };
        let mut free = BTreeSet::new();
        free.insert(region);
        Block { free }
    }

    fn find_region(&self, size: usize) -> Option<Region> {
        for free in &self.free {
            if size <= free.size() {
                return Some(free.clone());
            }
        }
        None
    }

    fn allocate(&mut self, size: usize) -> Option<Region> {
        self.find_region(size).map(|region| {
            self.free.remove(&region);
            let res = Region {
                start: region.start,
                end: region.start + size as u64,
            };
            if size < region.size() {
                let new = Region {
                    start: res.end,
                    end: region.end,
                };
                self.free.insert(new);
            }
            res
        })
    }

    fn before(&self, region: &Region) -> Option<Region> {
        self.free
            .range((Bound::Unbounded, Bound::Excluded(region)))
            .next_back()
            .cloned()
    }

    fn after(&self, region: &Region) -> Option<Region> {
        self.free
            .range((Bound::Excluded(region), Bound::Unbounded))
            .next()
            .cloned()
    }

    fn deallocate(&mut self, region: Region) {
        let mut region = region;
        if let Some(before) = self.before(&region) {
            if before.end == region.start {
                region.start = before.start;
                self.free.remove(&before);
            }
        }
        if let Some(after) = self.after(&region) {
            if region.end == after.start {
                region.end = after.end;
                self.free.remove(&after);
            }
        }
        self.free.insert(region);
    }

    #[cfg(test)]
    fn regions(&self) -> Vec<Region> {
        let mut res: Vec<_> = self.free.iter().cloned().collect();
        res.sort();
        res
    }
}

#[derive(Default)]
pub struct Allocator {
    blocks: HashMap<Value, Block>,
}

pub struct Fragment {
    pub base: Value,
    pub offset: u64,
    pub size: usize,
}

impl Allocator {
    pub fn allocate(&mut self, size: usize) -> Option<Fragment> {
        for (value, block) in self.blocks.iter_mut() {
            if let Some(region) = block.allocate(size) {
                return Some(Fragment {
                    base: *value,
                    offset: region.start,
                    size,
                });
            }
        }
        None
    }

    pub fn deallocate(&mut self, fragment: Fragment) {
        let block = self.blocks.get_mut(&fragment.base).unwrap();
        block.deallocate(Region {
            start: fragment.offset,
            end: fragment.offset + fragment.size as u64,
        });
    }

    pub fn append_block(&mut self, base: Value, size: usize, allocated: usize) -> Fragment {
        let block = Block::new_with_allocated(size, allocated);
        self.blocks.insert(base, block);
        Fragment {
            base,
            offset: 0,
            size: allocated,
        }
    }
}

#[test]
fn test_block() {
    let mut block = Block::new(1024);
    let f0 = block.allocate(128);
    let f1 = block.allocate(128);
    let f2 = block.allocate(128);

    assert_eq!(f0, Some(Region { start: 0, end: 128 }));
    assert_eq!(
        f1,
        Some(Region {
            start: 128,
            end: 256
        })
    );
    assert_eq!(
        f2,
        Some(Region {
            start: 256,
            end: 384
        })
    );

    block.deallocate(f2.unwrap());
    block.deallocate(f0.unwrap());
    assert_eq!(
        block.regions(),
        vec![
            Region { start: 0, end: 128 },
            Region {
                start: 256,
                end: 1024
            }
        ]
    );
    let f3 = block.allocate(128);
    assert_eq!(f3, Some(Region { start: 0, end: 128 }));
    assert_eq!(
        block.regions(),
        vec![Region {
            start: 256,
            end: 1024
        }]
    );
    block.deallocate(f3.unwrap());

    let f4 = block.allocate(512);
    assert_eq!(
        f4,
        Some(Region {
            start: 256,
            end: 768
        })
    );

    assert_eq!(block.allocate(1024), None);

    block.deallocate(f1.unwrap());
    block.deallocate(f4.unwrap());
    assert_eq!(
        block.regions(),
        vec![Region {
            start: 0,
            end: 1024
        }]
    );
}
