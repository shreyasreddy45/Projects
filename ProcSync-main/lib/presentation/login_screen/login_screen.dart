import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

import '../../core/app_export.dart';
import '../../widgets/custom_image_view.dart';
import './bloc/login_bloc.dart';
import 'package:ProcSync/presentation/chat/models/user_model.dart';
import 'package:ProcSync/presentation/dashboard_page/models/class_model.dart';

class LoginScreen extends StatelessWidget {
  const LoginScreen({Key? key}) : super(key: key);

  static Widget builder(BuildContext context) {
    return BlocProvider<LoginBloc>(
      create: (context) => LoginBloc()..add(LoginInitialEvent()),
      child: const LoginScreen(),
    );
  }

  @override
  Widget build(BuildContext context) {
    return BlocConsumer<LoginBloc, LoginState>(
      listener: (context, state) async {
        if (state.isAuthenticated) {
          final user = FirebaseAuth.instance.currentUser;
          if (user != null) {
            final userDoc =
                FirebaseFirestore.instance.collection('users').doc(user.uid);
            final docSnapshot = await userDoc.get();

            if (!docSnapshot.exists) {
              // Store base info from Google
              await userDoc.set({
                'uid': user.uid,
                'name': user.displayName,
                'email': user.email,
                'photoUrl': user.photoURL,
                'createdAt': FieldValue.serverTimestamp(),
              });
            }
          }

          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Login successful!')),
          );
          if (user != null) {
            final userDoc =
                FirebaseFirestore.instance.collection('users').doc(user.uid);
            final docSnapshot = await userDoc.get();

            if (!docSnapshot.exists) {
              await userDoc.set({
                'uid': user.uid,
                'name': user.displayName,
                'email': user.email,
                'photoUrl': user.photoURL,
                'createdAt': FieldValue.serverTimestamp(),
              });
              Navigator.pushReplacementNamed(context, AppRoutes.accScreen);
            } else {
              final data = docSnapshot.data()!;
              final role = data['role'];
              final joinedClassId = data['joinedClassId'];

              if (role == 'student') {
                if (joinedClassId == null || joinedClassId.isEmpty) {
                  Navigator.pushReplacementNamed(
                      context, AppRoutes.studentJoinPage);
                } else {
                  // Fetch the joined class and navigate to ChatPage
                  final classDoc = await FirebaseFirestore.instance
                      .collection('classes')
                      .doc(joinedClassId)
                      .get();

                  if (classDoc.exists) {
                    final classData = classDoc.data()!;
                    final classModel =
                        ClassModel.fromMap(classData, id: classDoc.id);

                    final userModel = UserModel(
                      uid: data['uid'],
                      name: data['name'],
                      email: data['email'],
                      role: data['role'],
                      joinedClassId: joinedClassId,
                    );

                    Navigator.pushReplacementNamed(
                      context,
                      AppRoutes.chatPage,
                      arguments: {
                        'user': userModel,
                        'classModel': classModel,
                      },
                    );
                  } else {
                    // class doesn't exist, fallback
                    Navigator.pushReplacementNamed(
                        context, AppRoutes.studentJoinPage);
                  }
                }
              } else if (role == 'proctor') {
                Navigator.pushReplacementNamed(
                    context, AppRoutes.dashboardPage);
              } else {
                Navigator.pushReplacementNamed(context, AppRoutes.accScreen);
              }
            }
          }
        } else if (state.errorMessage != null) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text(state.errorMessage!)),
          );
        }
      },
      builder: (context, state) {
        return Scaffold(
          backgroundColor: appTheme.whiteCustom,
          body: SafeArea(
            child: Center(
              child: Padding(
                padding: EdgeInsets.symmetric(horizontal: 24.h),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    CustomImageView(
                      imagePath: ImageConstant.imgScreenshot202504032248481,
                      height: 200.h,
                      width: 200.h,
                      margin: EdgeInsets.only(bottom: 64.h),
                    ),
                    _buildGoogleSignInButton(context, state),
                  ],
                ),
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildGoogleSignInButton(BuildContext context, LoginState state) {
    return GestureDetector(
      onTap: () {
        if (!state.isLoading) {
          context.read<LoginBloc>().add(GoogleSignInTappedEvent());
        }
      },
      child: Container(
        width: 288.h,
        height: 49.h,
        decoration: BoxDecoration(
          color: appTheme.color7FF5F1,
          borderRadius: BorderRadius.circular(12.h),
          boxShadow: [
            BoxShadow(
              color: appTheme.color1F0000,
              blurRadius: 9,
              offset: const Offset(-5, 5),
            ),
          ],
        ),
        child: Stack(
          alignment: Alignment.center,
          children: [
            if (state.isLoading)
              SizedBox(
                height: 24.h,
                width: 24.h,
                child: CircularProgressIndicator(
                  strokeWidth: 2.h,
                  valueColor:
                      const AlwaysStoppedAnimation<Color>(Colors.black54),
                ),
              )
            else
              Row(
                children: [
                  Padding(
                    padding: EdgeInsets.only(left: 16.h),
                    child: CustomImageView(
                      imagePath: ImageConstant.imgIconGoogleIcon,
                      height: 33.h,
                      width: 33.h,
                    ),
                  ),
                  Padding(
                    padding: EdgeInsets.only(left: 32.h),
                    child: Text(
                      "Continue with Google",
                      style: TextStyleHelper.instance.title16,
                    ),
                  ),
                ],
              ),
          ],
        ),
      ),
    );
  }
}
