#![no_std]
#![forbid(unsafe_code)]

use core::{
    cell::Cell,
    future::Future,
    marker::PhantomData,
    pin::Pin,
    sync::atomic::{AtomicU32, Ordering::Relaxed},
    task::{Context, Poll, Waker},
};

use futures::{pin_mut, task::noop_waker};

pub trait Mask: Copy {
    fn as_bits(self) -> u32;
}

pub struct Events<S> {
    events: AtomicU32,
    _phantom: PhantomData<S>,
}

pub struct Signals<'a, S> {
    pending: &'a Events<S>,
    run: Cell<Run>,
}

pub struct Executor<'a, S> {
    signals: &'a Signals<'a, S>,
    waker: Waker,
}

pub struct Park<'a> {
    pending: &'a AtomicU32,
    wakeup: u32,
}

pub struct Parked<'a> {
    last_pending: u32,
    wakeup: u32,
    _phantom: PhantomData<&'a ()>,
}

impl<S> Default for Events<S> {
    fn default() -> Self {
        Events {
            events: Default::default(),
            _phantom: PhantomData,
        }
    }
}

impl<S: Mask> Events<S> {
    pub fn raise(&self, signal: S) {
        self.events.fetch_or(signal.as_bits(), Relaxed);
    }
}

impl<'a, S: Mask> Signals<'a, S> {
    pub fn watch(pending: &'a Events<S>) -> Self {
        Signals {
            pending,
            run: Default::default(),
        }
    }

    pub async fn drive<T, E, F>(&self, signal: S, poll: F) -> Result<T, E>
    where
        F: Unpin + FnMut() -> nb::Result<T, E>,
    {
        (Drive {
            run: &self.run,
            mask: signal.as_bits(),
            poll,
        })
        .await
    }
}

impl<'a, S> Executor<'a, S> {
    pub fn bind(signals: &'a Signals<S>) -> Self {
        Executor {
            signals,
            waker: noop_waker(),
        }
    }

    pub fn with_waker(&mut self, waker: Waker) -> &mut Self {
        self.waker = waker;
        self
    }
}

impl<S> Executor<'_, S> {
    pub fn run_with_park<F, P>(&mut self, future: F, mut park: P) -> F::Output
    where
        F: Future,
        P: for<'a> FnMut(Park<'a>) -> Parked<'a>,
    {
        pin_mut!(future);

        let pending = &self.signals.pending.events;
        let mut cx = Context::from_waker(&self.waker);

        let (mut last_pending, mut wakeup) = (u32::MAX, u32::MAX);

        loop {
            if last_pending & wakeup != 0 {
                self.signals.run.set(Run {
                    raised: last_pending,
                    wakeup: 0,
                });

                match future.as_mut().poll(&mut cx) {
                    Poll::Ready(output) => break output,
                    Poll::Pending => (),
                }

                wakeup = self.signals.run.get().wakeup;
            }

            last_pending = park(Park { pending, wakeup }).last_pending;

            loop {
                let cleared = last_pending & !wakeup;
                match pending.compare_exchange_weak(last_pending, cleared, Relaxed, Relaxed) {
                    Ok(_) => break,
                    Err(current) => last_pending = current,
                }
            }
        }
    }
}

impl<'a> Park<'a> {
    pub fn race_free(self) -> Parked<'a> {
        Parked {
            last_pending: self.pending.load(Relaxed),
            wakeup: self.wakeup,
            _phantom: PhantomData,
        }
    }
}

impl Parked<'_> {
    pub fn is_idle(&self) -> bool {
        self.last_pending & self.wakeup == 0
    }
}

#[derive(Copy, Clone, Default)]
struct Run {
    raised: u32,
    wakeup: u32,
}

struct Drive<'a, F> {
    run: &'a Cell<Run>,
    mask: u32,
    poll: F,
}

impl<T, E, F> Future for Drive<'_, F>
where
    F: Unpin + FnMut() -> nb::Result<T, E>,
{
    type Output = Result<T, E>;

    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context) -> Poll<Self::Output> {
        let run = self.run.get();

        if run.raised & self.mask != 0 {
            match (self.poll)() {
                Ok(ok) => return Poll::Ready(Ok(ok)),
                Err(nb::Error::Other(err)) => return Poll::Ready(Err(err)),
                Err(nb::Error::WouldBlock) => (),
            }
        }

        let wakeup = run.wakeup | self.mask;
        self.run.set(Run { wakeup, ..run });

        Poll::Pending
    }
}
