// jsdom has no IntersectionObserver; Framer Motion's whileInView needs one. Report everything as in view.
class InView {
  constructor(private cb: (e: { isIntersecting: boolean; target: Element }[]) => void) {}
  observe(target: Element) {
    this.cb([{ isIntersecting: true, target }]);
  }
  unobserve() {}
  disconnect() {}
}
(globalThis as unknown as { IntersectionObserver: unknown }).IntersectionObserver = InView;
