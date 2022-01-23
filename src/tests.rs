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
        let wait = signals.drive_infallible(Signal::B, || {
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
        .block_on(future, |park| {
            let parked = park.race_free();
            assert!(!parked.is_idle());
            parked
        });
}

bitflags! {
    struct Signal: u32 {
        const A = 1 << 3;
        const B = 1 << 11;
        const C = 1 << 17;
        const ALL = Self::A.bits | Self::B.bits | Self::C.bits;
    }
}

impl EventMask for Signal {
    fn as_bits(self) -> u32 {
        self.bits()
    }
}

impl task::ArcWake for Events<Signal> {
    fn wake_by_ref(arc_self: &Arc<Self>) {
        arc_self.raise(Signal::ALL)
    }
}
