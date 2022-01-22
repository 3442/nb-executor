# `nb-executor`

This crate provides an async executor with the following features:

- `#![no_std]`, does not depend on `alloc`.

- `#![forbid(unsafe_code)]`.

- Runs a single future on the current thread of execution. Concurrency can be
  achieved with multiplexers such as [`futures::join!()`][futures-join]. The
  executor itself is `!Send + !Sync`.

- A `Sync`-friendly, atomic 32-bit *event mask* is shared between the executor
  and all external event sources. Users define an event set (with, for example,
  [`bitflags`][bitflags]) to make the event mask meaningful.

- Wakeups correspond to edge-triggered *signals*. A shared reference to the
  event mask is sufficient to *raise* a signal.

- Async code can *drive* non-blocking *poll functions* with a given *signal
  mask*. The poll function is attempted whenever any of the signals in the mask
  is raised until it resolves to its output. This is the primary mechanism for
  performing asynchronous work.

- The future is polled only if an intersection exists between the current event
  mask and the combined signal mask. Likewise, a poll function won't be
  attempted if there is no intersection between its signal mask and a snapshot
  of the event mask taken just before polling the "root" future.

- When further progress is presumed not to be currently possible, a
  user-provided *park function* is invoked. This function may enter an
  event-free context and then verify whether there is in fact no work to do
  right now. it may then suspend the executor in some particular way, such as a
  WFI/WFE-style operation on embedded firmware or `std::thread::park()` on
  hosted systems.

- A no-op waker is used by default. It can be replaced for another one that
  cooperates with the park function.

[futures-join]: <https://docs.rs/futures/latest/futures/macro.join.html>
[bitflags]: <https://crates.io/crates/bitflags>

## License

`nb-executor` is licensed under Apache-2.0.

### Contributions

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in `nb-executor` by you shall be under the terms and conditions of
the Apache-2.0 license, without any additional terms or conditions.
