use super::*;
use bitflags::bitflags;
use futures::task;
use std::{sync::Arc, sync::Once};

#[test]
fn waker() {
    let events = Arc::new(Events::default());
    let signals = events.watch();

    let once = Once::new();
    let waker = task::waker(Arc::clone(&events));

    let future = async {
        let wait = signals.drive_infallible(Ev::B, || {
            once.is_completed().then(|| ()).ok_or(nb::Error::WouldBlock)
        });

        let wake = async {
            once.call_once(|| ());
            waker.wake_by_ref();
        };

        let ((), ()) = futures::join!(wait, wake);
    };

    signals
        .bind()
        .with_waker(waker.clone())
        .block_on(future, park_test);
}

#[cfg(feature = "heapless")]
#[test]
fn queue() {
    let events = Events::default();
    let signals = events.watch();

    let queue = Mpmc::<_, 4>::new();

    let producer = async {
        for n in 0..32 {
            queue.enqueue(n, &signals, Ev::A).await;
        }
    };

    let consumer = async {
        for n in 0..32 {
            assert_eq!(queue.dequeue(&signals, Ev::A).await, n);
        }

        assert_eq!(queue.inner().dequeue(), None);
    };

    let future = async { futures::join!(producer, consumer) };

    signals.bind().block_on(future, park_test);
}

fn park_test(park: Park<'_>) -> Parked {
    let parked = park.race_free();
    assert!(!parked.is_idle());
    parked
}

bitflags! {
    struct Ev: u32 {
        const A = 1 << 3;
        const B = 1 << 11;
        const C = 1 << 17;
        const ALL = Self::A.bits | Self::B.bits | Self::C.bits;
    }
}

impl EventMask for Ev {
    fn as_bits(self) -> u32 {
        self.bits()
    }
}

impl task::ArcWake for Events<Ev> {
    fn wake_by_ref(arc_self: &Arc<Self>) {
        arc_self.raise(Ev::ALL)
    }
}
