//! Single-future, `#![no_std]` executor based on event bitmasks.
//!
//! See `README.md` for a brief overview. [`Executor`] describes the most important elements
//! to get started.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

use core::{
    cell::Cell,
    convert::Infallible,
    future::Future,
    marker::PhantomData,
    pin::Pin,
    sync::atomic::{AtomicU32, Ordering::Relaxed},
    task::{Context, Poll, Waker},
};

pub use futures::{self, pin_mut};
pub use nb;

#[cfg(test)]
mod tests;

/// Definition of event bits.
///
/// Implementors should be simple wrappers around `u32` that allow application code to clearly
/// differentiate the relevant event sources. Events and signals correspond to the bits of the
// value returned by `as_bits()`.
///
/// # Examples
///
/// ```
/// # use nb_executor::EventMask;
/// use bitflags::bitflags;
///
/// bitflags! {
///     struct Ev: u32 {
///         const USB_RX = 1 << 0;
///         const PRIO = 1 << 1;
///         const TICK = 1 << 2;
///     }
/// }
///
/// impl EventMask for Ev {
///     fn as_bits(self) -> u32 {
///         self.bits()
///     }
/// }
/// ```
pub trait EventMask: Copy {
    /// Get the `u32` bitmask.
    fn as_bits(self) -> u32;
}

/// Shared event mask.
///
/// The event mask is an atomic bitmask shared between the executor and the event sources. The type
/// parameter `S` is an [`EventMask`].
///
/// # Events or signals?
/// They are the same. The name "event" refers to their presence in the shared event mask,
/// while the name "signal" refers to poll-local signal masks. Events become signals when they
/// are handled or waited for.
pub struct Events<S> {
    events: AtomicU32,
    _phantom: PhantomData<S>,
}

/// Signal state manager and event listener.
///
/// `Signals` maintains the signal state of an [`Executor`]. This consists of the raised signal set
/// and the wakeup signal set. Upon polling the future, the raised signal set is frozen to the
/// then-current value of the event mask, all bits in the wakeup signal set are removed from the
/// event mask (atomically), and the wakeup signal set is cleared. See also [`Step::poll()`]
/// for behavior when the wakeup signal set is zero. Current poll functions which are driven
/// by any of the raised signals will be attempted. If a poll function is not attempted or does
/// not yet resolve to an output then its signal mask will be OR-ed into the wakeup signal set.
/// The future won't be polled again until a raised signal matches the wakeup signal set.
///
/// The type parameter `S` is an [`EventMask`].
///
/// # Delayed signals
/// The executor will examine the event mask after polling the future to check for any immediate
/// updates. Since the raised signal set remains frozen during polling, any signal raised by the
/// future itself won't become visible until this happens. This also means that external events are
/// not tested for until the future yields. Correct programs are not able to observe this behavior.
pub struct Signals<'a, S> {
    pending: &'a Events<S>,
    run: Cell<Run>,
}

/// A single-threaded, single-future async executor.
///
/// # Typical setup
/// - First, an event mask is created with [`Events::default()`] (or [`Events::new()`] if
///   that needs to be `const`). Event masks are Send + Sync` and shared references to the
///   are enough for all operations, so they work well as `static` items or other types of
///   shared state. Distribute event mask references to the external event sources and keep
///   one for the executor.
///
/// - The event mask is then watched with [`Events::watch()`]. The resulting [`Signals`] is
///   `Send + !Sync`. This means that operations become limited to one thread of execution
///   from this point on, so this is usually done in some type of initialization or main
///   function instead of in a shared or global context.
///
/// - A new executor is bound with [`Signals::bind()`]. Executors are `!Send + !Sync`:
///   neither it nor the associated `Signals` may escape the current thread. This makes them
///   appropriate for construction at the use site.
///
/// - A future is created. It needs a reference to the `Signals` object in order to drive
///   poll functions, making it `!Sync` too.
///
/// - Finally, [`Executor::block_with_park()`] blocks and resolves the future while external
///   event sources direct it through the event mask, possibly with help from the park function.
///
/// # Examples
///
/// This is a complete usage example. It uses `std::sync` primitives and a park function based on
/// `std::thread::park()` to multiply the integers from 1 to 10 read from a blocking queue.
///
/// ```
/// # use nb_executor::*;
/// # use bitflags::bitflags;
/// use std::{thread, sync::{mpsc::*, Arc}};
///
/// bitflags! {
///     struct Ev: u32 {
///         const QUEUE = 1 << 0;
///     }
/// }
///
/// impl EventMask for Ev {
///     fn as_bits(self) -> u32 {
///         self.bits()
///     }
/// }
///
/// async fn recv(signals: &Signals<'_, Ev>, rx: &Receiver<u32>) -> Option<u32> {
///     signals.drive_infallible(Ev::QUEUE, || match rx.try_recv() {
///         Ok(n) => Ok(Some(n)),
///         Err(TryRecvError::Disconnected) => Ok(None),
///         Err(TryRecvError::Empty) => Err(nb::Error::WouldBlock),
///     }).await
/// }
///
/// let events = Arc::new(Events::default());
/// let signals = events.watch();
///
/// let (tx, rx) = sync_channel(1);
/// let future = async {
///     let mut product = 1;
///     while let Some(n) = recv(&signals, &rx).await {
///         product *= n;
///     }
///
///     product
/// };
///
/// let events_prod = Arc::clone(&events);
/// let runner = thread::current();
///
/// thread::spawn(move || {
///     for n in 1..=10 {
///         tx.send(n).unwrap();
///         events_prod.raise(Ev::QUEUE);
///         runner.unpark();
///     }
///
///     // Notify shutdown
///     drop(tx);
///     events_prod.raise(Ev::QUEUE);
///     runner.unpark();
/// });
///
/// let result = signals.bind().block_with_park(future, |park| {
///     // thread::park() is event-safe, no lock is required
///     let parked = park.race_free();
///     if parked.is_idle() {
///         thread::park();
///     }
///
///     parked
/// });
///
/// assert_eq!(result, (1..=10).product()); // 3628800
/// ```
pub struct Executor<'a> {
    pending: &'a AtomicU32,
    run: &'a Cell<Run>,
    waker: Waker,
}

/// A non-blocking executor-future-park state machine.
///
/// All executor operations are ultimately implemented with `Step`. It allows fine-grained
/// control over execution and control flow. `Step` can only perform one poll step at a time
/// and requires a pinned future. A `Step` object is created by calling [`Executor::step()`].
///
/// See [`Executor::block_with_park()`] for a blocking runner.
pub struct Step<'exec, 'fut, F, P> {
    executor: Executor<'exec>,
    wakeup: u32,
    last_pending: u32,
    future: Pin<&'fut mut F>,
    park: P,
}

/// A request to park the executor.
///
/// Parking is the mechanism by which the executor *tries* to wait for external event sources when
/// no signal in the wakeup set is currently raised (see [`Signals`]). The executor might
/// nonetheless resume immediately if a signal is raised before the atomic part of the *park
/// protocol* takes place. Park functions implement the park protocol and must follow it strictly,
/// **you risk deadlocks otherwise**.
///
/// # The park protocol
///
/// - First, the executor determines that further progress is unlikely at this moment. The
///   specifics of this process are implementation details that should not be relied upon.
///
/// - The park function is called with a `Park` parameter.
///
/// - The park function enters a context wherein no external events may influence a correct
///   decision to sleep or not. For example, a park function that does not sleep at all does
///   not need to do anything here, since no external event can incorrectly change that
///   behavior. On the other hand, a park function that halts until a hardware interrupt occurs
///   would need to enter an interrupt-free context to avoid deadlocks.
///
/// - The park function calls [`Park::race_free()`] while still in the event-safe context.
///   This produces a [`Parked`] value that serves as proof of the call to `race_free()`.
///
/// - If the park function intends to block or sleep, then it must first call
///   [`Parked::is_idle()`]. It may be allowed to sleep only if that function returns `true`.
///
/// - If the park function is willing to sleep and is allowed to do so, it must
///   atomically exit the event-safe context whilst entering the sleep state. A deadlock is
///   again possible if both operations are not done atomically with respect to each other
///   (**only for blocking runners, see below**).
///
/// - If the park function sleeps, this state should be automatically exited when an external
///   event occurs.
///
/// - The park function returns its [`Parked`] token.
///
/// - The executor resumes.
///
/// # Delegating out of the park function
///
/// Blocking runners, such as [`Executor::block_with_park()`], require that the park function's
/// event-safe context be exited in an atomic manner with respect to the start of whatever
/// blocking operation. However, this requirement does not hold for [`Step`] **as long as
/// the park function never exits the event-safe context and it itself never blocks**. Since
/// [`Step::poll()`] will return after parking, the caller can perform potentially-blocking
/// operations from the event-safe context. It must still release it atomically if it will
/// sleep, though. With this technique it is even possible for the `poll()` caller to be an
/// external event source.
pub struct Park<'a> {
    pending: &'a AtomicU32,
    wakeup: u32,
}

/// Proof of parking.
///
/// Park functions return `Parked` objects as a proof of having called [`Park::race_free()`]. This
/// is necessary because [`Park::race_free()`] updates executor state and must always be run.
/// `Parked` can be used by park functions to determine whether blocking or sleeping is
/// permissible. See [`Park`] documentation for the correct parking protocol.
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

impl<S> Events<S> {
    /// Construct a new event mask.
    ///
    /// This is the same as [`Events::default()`] but is also `const`.
    pub const fn new() -> Self {
        Events {
            events: AtomicU32::new(0),
            _phantom: PhantomData,
        }
    }

    /// Associate a new `Signals` to this event mask.
    ///
    /// This indirectly links the executor to the event mask, since creating an [`Executor`]
    /// requires a `Signals`.
    #[must_use]
    pub fn watch(&self) -> Signals<S> {
        Signals {
            pending: self,
            run: Default::default(),
        }
    }
}

impl<S: EventMask> Events<S> {
    /// Raise all events in a mask.
    ///
    /// This operation is atomic: multiple events can be safely raised at the same time.
    /// Already raised signals are left as is.
    pub fn raise(&self, signals: S) {
        self.events.fetch_or(signals.as_bits(), Relaxed);
    }
}

impl<'a, S: EventMask> Signals<'a, S> {
    /// Bind a new executor to this signal source.
    ///
    /// The executor will be constructed with a waker that does nothing. It can be replaced
    /// by calling [`Executor::with_waker()`].
    #[must_use]
    pub fn bind(&self) -> Executor {
        Executor {
            pending: &self.pending.events,
            run: &self.run,
            waker: futures::task::noop_waker(),
        }
    }

    /// Asynchronously drive a fallible poll function to completion.
    ///
    /// The future produced by this method checks on each poll whether any of the signals
    /// in `signals` is present in the raised signal set. If that is the case, it invokes
    /// the poll function. If the poll function succeeds or fails with [`nb::Error::Other`],
    /// the future completes immediately with that value. If the poll function returns
    /// [`nb::Error::WouldBlock`], or if none of the signals in `signals` is present in the
    /// raised signal sent, then `signals` is added to the wakeup signal set and the
    /// future pends.
    ///
    /// `poll()` must handle spurious calls gracefully. There is no guarantee that any of
    /// the intended effects of any signal in `signals` has actually taken place. `poll()`
    /// may not block.
    pub async fn drive<T, E, F>(&self, signals: S, mut poll: F) -> Result<T, E>
    where
        F: FnMut() -> nb::Result<T, E>,
    {
        let mask = signals.as_bits();

        futures::future::poll_fn(move |_| {
            let run = self.run.get();

            if run.raised & mask != 0 {
                match poll() {
                    Ok(ok) => return Poll::Ready(Ok(ok)),
                    Err(nb::Error::Other(err)) => return Poll::Ready(Err(err)),
                    Err(nb::Error::WouldBlock) => (),
                }
            }

            let wakeup = run.wakeup | mask;
            self.run.set(Run { wakeup, ..run });

            Poll::Pending
        })
        .await
    }

    /// Asynchronously drive an infallible poll function to completion.
    ///
    /// This is a variant of [`Signals::drive()`] intended for cases where there is no
    /// proper error type. Although `drive(sig, poll).await.unwrap()` works the same, it
    /// often requires explicit type annotations if `poll` is a closure. This method should
    /// be preferre in such cases, as well as when `poll` is a wrapper around
    /// `option.ok_or(WouldBlock)`.
    pub async fn drive_infallible<T, F>(&self, signals: S, poll: F) -> T
    where
        F: FnMut() -> nb::Result<T, Infallible>,
    {
        self.drive(signals, poll).await.unwrap()
    }
}

impl<'exec> Executor<'exec> {
    /// Replace the executor's waker with a custom one.
    ///
    /// No restrictions are imposed on the waker: `nb-executor` does not use wakers at all.
    /// Application code may define some communication between it and the park function.
    #[must_use]
    pub fn with_waker(mut self, waker: Waker) -> Self {
        self.waker = waker;
        self
    }

    /// Begin stepped execution.
    ///
    /// This allows the caller to remain in control of program flow in between polls, unlike
    /// [`Executor::block_with_park()`]. The future is run as specified in [`Step`] and
    /// [`Signals`]. `park` is a park function and must follow the [`Park`] protocol. The
    /// caller and the park function may cooperate to block or sleep outside of the park
    /// function within the protocol requirements.
    pub fn step<'fut, F, P>(self, future: Pin<&'fut mut F>, park: P) -> Step<F, P>
    where
        F: Future,
        P: FnMut(Park) -> Parked,
        'exec: 'fut,
    {
        Step {
            executor: self,
            wakeup: 0,
            last_pending: u32::MAX,
            future,
            park,
        }
    }

    /// Execute a future on this executor, parking when no progress is possible.
    ///
    /// This method will block until the future resolves. There are two possible states of
    /// operation while the future is executed:
    ///
    /// - Polling: The future's `poll()` method is called in order to attempt to resolve
    ///   it. The signal state is prepared as documented in [`Signals`] when switching
    ///   to the polling state. The next state after polling is unspecified, but will
    ///   eventually lead to parking if the future pends consistently.
    ///
    /// - Parking: This state is entered when useful work is unlikely at the current
    ///   time. For details, see the parking protocol in [`Park`]. `park` must adhere to
    ///   this protocol.
    ///
    /// See also [`Step`] and [`Executor::step()`].
    pub fn block_with_park<F, P>(self, future: F, park: P) -> F::Output
    where
        F: Future,
        P: FnMut(Park) -> Parked,
    {
        pin_mut!(future);
        let mut step = self.step(future, park);

        loop {
            match step.poll() {
                Poll::Pending => continue,
                Poll::Ready(ready) => break ready,
            }
        }
    }

    /// Execute a future in a busy-waiting loop.
    ///
    /// This is equivalent to calling [`Executor::block_with_park()`] with a park function that
    /// never sleeps. This is most likely the wrong way to do whatever you intend, prefer to
    /// define a proper wake function.
    pub fn block_busy<F: Future>(self, future: F) -> F::Output {
        self.block_with_park(future, |park| park.race_free())
    }
}

impl<F, P> Step<'_, '_, F, P>
where
    F: Future,
    P: FnMut(Park) -> Parked,
{
    /// Perform at most one non-blocking polling attempt.
    ///
    /// This method never blocks unless the park function does (or the future, but that's
    /// totally wrong anyway). The future will be polled given any of these:
    ///
    /// - The wakeup signal set is zero. This is intended to provide a reset or "full scan"
    ///   mechanism. In this case, the raised signal set will be hard-coded to all-ones,
    ///   regardless of the current value of the event mask. **Note:** A side effect of this
    ///   special case is that [`core::future::pending()`] and similar constructs might
    ///   busy-loop depending on the particular park function.
    ///
    /// - Any bit in the wakeup signal set matches the event mask. The event mask is
    ///   atomically frozen at the same time that this condition is check, creating the
    ///   raised signal set.
    ///
    /// The wakeup signal set is cleared just before polling the future (edge-triggering).
    /// Regardless of the poll result or if the future is polled at all, a park
    /// operation may be triggered under implementation-defined conditions. If parking
    /// occurs, it will be the last observable operation before `poll()` returns. The
    /// return value will be [`Poll::Pending`] if the future is not polled. Do not call
    /// call `poll()` again after it returns [`Poll::Ready`].
    ///
    /// On first call, the wakeup signal set is zero. This ensures that all driven poll
    /// functions are attempted at least once.
    pub fn poll(&mut self) -> Poll<F::Output> {
        let woken = if self.wakeup == 0 {
            self.last_pending = u32::MAX;
            true
        } else {
            loop {
                let cleared = self.last_pending & !self.wakeup;
                let result = self.executor.pending.compare_exchange_weak(
                    self.last_pending,
                    cleared,
                    Relaxed,
                    Relaxed,
                );

                match result {
                    Ok(current) => break cleared != current,
                    Err(current) => self.last_pending = current,
                }
            }
        };

        if !woken {
            return Poll::Pending;
        }

        self.executor.run.set(Run {
            raised: self.last_pending,
            wakeup: 0,
        });

        let mut cx = Context::from_waker(&self.executor.waker);
        let poll = self.future.as_mut().poll(&mut cx);

        if poll.is_pending() {
            self.wakeup = self.executor.run.get().wakeup;
            let park = Park {
                pending: self.executor.pending,
                wakeup: self.wakeup,
            };

            self.last_pending = (self.park)(park).last_pending;
        }

        poll
    }
}

impl<'a> Park<'a> {
    /// Promise that new events won't race with sleeps and get a proof of parking.
    ///
    /// Park functions must call this method to obtain the [`Parked`] object that they
    /// return, which also allows them to determine sleep permissibility. The caller
    /// promises that external events which may occur from the start of this call until
    /// optionally starting to sleep won't result in race conditions.
    pub fn race_free(self) -> Parked<'a> {
        Parked {
            last_pending: self.pending.load(Relaxed),
            wakeup: self.wakeup,
            _phantom: PhantomData,
        }
    }
}

impl Parked<'_> {
    /// Check whether useful work is certainly not possible until an event is raised.
    ///
    /// Unlike the calling of the park function, this is not an optimistic operation.
    /// Its result will be exact as long as the [`Park`] protocol is correctly followed.
    /// A return value of `false` prohibits the park function from sleeping at all:
    /// it should yield control to the executor immediately. A return value of `true`
    /// is a strong hint to sleep, block, or otherwise take over control flow until
    /// some unspecified condition, ideally until an event is raised.
    ///
    /// See also the [`Park`] protocol.
    pub fn is_idle(&self) -> bool {
        self.last_pending & self.wakeup == 0
    }
}

#[derive(Copy, Clone, Default)]
struct Run {
    raised: u32,
    wakeup: u32,
}
