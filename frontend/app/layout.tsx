import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'RiskShield — AI Risk Manager',
  description: 'Rupee-optimal chargeback, return, and abuse-ring defense for Indian merchants. Real-time scoring with measured precision, recall, and false-positive cost.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
