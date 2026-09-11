/**
 * Purpose : Big high-contrast text of what the app said and what it heard — for low-vision users and the demo.
 * Depends : nothing.
 * Question you should be able to answer: why is aria-live="polite" on the caption?
 */
export function Caption({ said, heard }: { said: string; heard: string }) {
  return (
    <div className="pointer-events-none absolute inset-x-0 top-0 p-4 text-white">
      <p aria-live="polite" className="text-3xl font-bold leading-tight drop-shadow-[0_2px_4px_rgba(0,0,0,.9)]">{said}</p>
      <p className="mt-2 text-lg text-yellow-300 drop-shadow-[0_2px_4px_rgba(0,0,0,.9)]">{heard}</p>
    </div>
  );
}
