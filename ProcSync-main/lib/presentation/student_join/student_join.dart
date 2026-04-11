import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'bloc/student_join_bloc.dart';
import 'package:ProcSync/routes/app_routes.dart';

class StudentJoinPage extends StatefulWidget {
  const StudentJoinPage({super.key});

  static Widget builder(BuildContext context) {
    return BlocProvider(
      create: (_) => StudentJoinBloc(),
      child: const StudentJoinPage(),
    );
  }

  @override
  State<StudentJoinPage> createState() => _StudentJoinPageState();
}

class _StudentJoinPageState extends State<StudentJoinPage> {
  final GoogleSignIn _googleSignIn = GoogleSignIn();
  GoogleSignInAccount? _currentUser;

  @override
  void initState() {
    super.initState();
    _handleSignInSilently();
  }

  Future<void> _handleSignInSilently() async {
    final user = await _googleSignIn.signInSilently();
    if (mounted) {
      setState(() {
        _currentUser = user;
      });
    }
  }

  Future<void> _handleSwitchAccount() async {
    await _googleSignIn.signOut();
    final user = await _googleSignIn.signIn();
    if (mounted && user != null) {
      setState(() {
        _currentUser = user;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final bloc = context.read<StudentJoinBloc>();
    final displayName = _currentUser?.displayName ?? "Student";
    final email = _currentUser?.email ?? "student@example.com";
    final photoUrl = _currentUser?.photoUrl;

    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: BlocConsumer<StudentJoinBloc, StudentJoinState>(
          listener: (context, state) {
            if (state.status == JoinStatus.success &&
                state.userModel != null &&
                state.classModel != null) {
              Navigator.pushNamedAndRemoveUntil(
                context,
                AppRoutes.chatPage,
                (route) => false,
                arguments: {
                  'user': state.userModel!,
                  'classModel': state.classModel!,
                },
              );
              ;
            } else if (state.status == JoinStatus.failure) {
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(
                  content: Text(state.errorMessage ?? 'Failed to join'),
                  backgroundColor: Colors.red,
                ),
              );
            }
          },
          builder: (context, state) {
            return SingleChildScrollView(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Column(
                children: [
                  const SizedBox(height: 30),

                  // Logo
                  Image.asset(
                    'assets/images/logo_full.png',
                    width: MediaQuery.of(context).size.width * 0.7,
                    height: 100,
                    fit: BoxFit.contain,
                  ),

                  const SizedBox(height: 20),

                  // User Card
                  Card(
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    elevation: 2,
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Column(
                        children: [
                          const Text("You're currently signed in as"),
                          const SizedBox(height: 12),
                          photoUrl != null
                              ? CircleAvatar(
                                  radius: 25,
                                  backgroundImage: NetworkImage(photoUrl),
                                )
                              : CircleAvatar(
                                  radius: 25,
                                  backgroundColor: Colors.purple,
                                  child: Text(
                                    displayName[0],
                                    style: const TextStyle(color: Colors.white),
                                  ),
                                ),
                          const SizedBox(height: 8),
                          Text("$displayName\n$email",
                              textAlign: TextAlign.center),
                          const SizedBox(height: 8),
                          GestureDetector(
                            onTap: _handleSwitchAccount,
                            child: const Text(
                              "Switch account",
                              style: TextStyle(
                                color: Colors.blue,
                                decoration: TextDecoration.underline,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),

                  const SizedBox(height: 30),

                  const Text(
                    "Ask your proctor for the code, then enter it here",
                    style: TextStyle(fontSize: 16),
                  ),
                  const SizedBox(height: 10),

                  // Class Code Input
                  TextField(
                    onChanged: (value) =>
                        bloc.add(ClassCodeChanged(value.trim())),
                    decoration: InputDecoration(
                      hintText: "Enter code",
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                  ),

                  const SizedBox(height: 20),

                  // Join Button
                  ElevatedButton(
                    onPressed: () {
                      if (state.classCode.isNotEmpty) {
                        bloc.add(JoinClassSubmitted());
                      } else {
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(
                            content: Text("Please enter a class code"),
                          ),
                        );
                      }
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF6A1B9A),
                      padding: const EdgeInsets.symmetric(
                          horizontal: 24, vertical: 12),
                    ),
                    child: const Text(
                      "Join",
                      style: TextStyle(color: Colors.white, fontSize: 16),
                    ),
                  ),

                  const SizedBox(height: 30),

                  // Help Info
                  const Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text("To sign in with a code:"),
                      SizedBox(height: 8),
                      Text("• Use an authorized account"),
                      Text("• Use a class code with 6–8 letters or numbers"),
                      Text("  and no spaces or symbols"),
                    ],
                  ),
                  const SizedBox(height: 40),
                ],
              ),
            );
          },
        ),
      ),
    );
  }
}
