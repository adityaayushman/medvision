import { AuthProvider } from "@/components/dashboard/AuthContext";

export default function DashboardRootLayout({ children }: { children: React.ReactNode }) {
  return <AuthProvider>{children}</AuthProvider>;
}
