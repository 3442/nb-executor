use super::{EventMask, Signals};
use core::convert::Infallible;
use heapless::mpmc::MpMcQueue;
use nb::Error::WouldBlock;

/// Multi-producer, multi-consumer, fixed-capacity async queue.
///
/// This wraps a [`MpMcQueue`] from the `heapless` crate, adding support for
/// asynchronous enqueue and dequeue operations. `N` must be a power of two.
#[cfg_attr(docsrs, doc(cfg(feature = "heapless")))]
pub struct Mpmc<T, const N: usize>(MpMcQueue<T, N>);

impl<T, const N: usize> Default for Mpmc<T, N> {
    fn default() -> Self {
        Mpmc(Default::default())
    }
}

impl<T, const N: usize> Mpmc<T, N> {
    /// Creates an empty queue.
    pub const fn new() -> Self {
        Mpmc(MpMcQueue::new())
    }

    /// Accesses the inner [`MpMcQueue`].
    pub fn inner(&self) -> &MpMcQueue<T, N> {
        &self.0
    }

    /// Adds an item to the queue.
    ///
    /// If the queue is full, this will wait on `ev` until a successful insertion.
    /// After inserting the item, `ev` will be raised in order to signal producers.
    pub async fn enqueue<S: EventMask>(&self, item: T, signals: &Signals<'_, S>, ev: S) {
        let mut item = Some(item);
        let queued = signals.drive_infallible(ev, move || {
            self.try_enqueue(item.take().ok_or(WouldBlock)?, signals, ev)
                .map_err(|returned| {
                    item = Some(returned);
                    WouldBlock
                })
        });

        queued.await
    }

    /// Attempts to add an item to the queue.
    ///
    /// If the queue is currently full, the item is returned in the `Err` variant.
    /// See also [`Mpmc::enqueue()`].
    pub fn try_enqueue<S: EventMask>(
        &self,
        item: T,
        signals: &Signals<'_, S>,
        ev: S,
    ) -> Result<(), T> {
        let result = self.0.enqueue(item);
        if let Ok(()) = result {
            signals.pending().raise(ev);
        }

        result
    }

    /// Removes an item from the queue.
    ///
    /// If the queue is empty, this will wait on `ev` until an item is enqueued.
    /// After removing an item, `ev` will be raised to notify producers of available
    /// space in the queue.
    pub async fn dequeue<S: EventMask>(&self, signals: &Signals<'_, S>, ev: S) -> T {
        signals
            .drive_infallible(ev, || self.try_dequeue(signals, ev))
            .await
    }

    /// Attempts to remove an item from the queue.
    ///
    /// This will return `WouldBlock` if the queue is empty. See also [`Mpmc::dequeue()`].
    pub fn try_dequeue<S: EventMask>(
        &self,
        signals: &Signals<'_, S>,
        ev: S,
    ) -> nb::Result<T, Infallible> {
        let item = self.0.dequeue().ok_or(WouldBlock)?;

        signals.pending().raise(ev);
        Ok(item)
    }
}

impl<T, const N: usize> From<MpMcQueue<T, N>> for Mpmc<T, N> {
    fn from(queue: MpMcQueue<T, N>) -> Self {
        Mpmc(queue)
    }
}

impl<T, const N: usize> From<Mpmc<T, N>> for MpMcQueue<T, N> {
    fn from(queue: Mpmc<T, N>) -> Self {
        queue.0
    }
}
